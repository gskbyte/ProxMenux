"""Identify top CPU consumers (VMs, LXC containers, and host processes).

Reads /proc, /sys/fs/cgroup, /run/qemu-server, and /etc/pve directly with zero
subprocesses so high-CPU alerts can pinpoint what caused a sustained spike
without adding load or spawning tools on an already stressed node.

Security: never reads /proc/<pid>/cmdline or /proc/<pid>/environ so secrets in
command-line arguments can never leak into external notifications or AI prompts.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_CTRL_CHARS = re.compile(r'[\x00-\x1f\x7f]+')
_WHITESPACE = re.compile(r'\s+')
_QEMU_CGROUP_RE = re.compile(r'(?:^|/)qemu\.slice/(\d+)\.scope(?:/|$)')
_LXC_CGROUP_RE = re.compile(
    r'(?:^|/)(?:lxc/(\d+)|lxc\.payload\.(\d+)|lxc\.monitor\.(\d+))(?:/|$)'
)
_QEMU_SCOPE_DIR_RE = re.compile(r'^(\d+)\.scope$')


def _sanitize_label(value: Any, max_len: int = 48) -> str:
    """Strip control characters, collapse whitespace, and bound length."""
    if value is None:
        return ''
    text = _CTRL_CHARS.sub(' ', str(value))
    text = _WHITESPACE.sub(' ', text).strip()
    if len(text) > max_len:
        text = text[:max_len - 1].rstrip() + '…'
    return text


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b''


def _parse_proc_stat(raw: bytes) -> Optional[Tuple[str, int, int]]:
    """Return (comm, starttime_ticks, cpu_ticks) from /proc/<pid>/stat bytes."""
    lpar = raw.find(b'(')
    rpar = raw.rfind(b')')
    if lpar < 0 or rpar <= lpar:
        return None
    comm = raw[lpar + 1:rpar].decode('utf-8', errors='replace')
    rest = raw[rpar + 2:].split()
    if len(rest) <= 19:
        return None
    try:
        utime = int(rest[11])
        stime = int(rest[12])
        starttime = int(rest[19])
    except ValueError:
        return None
    return comm, starttime, utime + stime


def classify_cgroup(cgroup_text: str) -> Tuple[str, Optional[int]]:
    """Classify a /proc/<pid>/cgroup payload into ('vm'|'lxc'|'host', vmid)."""
    if not cgroup_text:
        return 'host', None
    for line in cgroup_text.splitlines():
        parts = line.split(':', 2)
        path = parts[2].strip() if len(parts) == 3 else line.strip()
        if not path:
            continue
        vm_match = _QEMU_CGROUP_RE.search(path)
        if vm_match:
            return 'vm', int(vm_match.group(1))
        lxc_match = _LXC_CGROUP_RE.search(path)
        if lxc_match:
            vmid = next((g for g in lxc_match.groups() if g), None)
            if vmid is not None:
                return 'lxc', int(vmid)
    return 'host', None


def _parse_cgroup_usage_usec(text: str) -> Optional[int]:
    for line in text.splitlines():
        if line.startswith('usage_usec '):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    return None


def _parse_conf_name(text: str, key_name: str) -> str:
    """Read top-level `name:` or `hostname:` from a PVE guest config."""
    prefix = f'{key_name}:'
    for line in text.splitlines():
        if line.startswith('['):
            break
        if line.startswith(prefix):
            return _sanitize_label(line[len(prefix):].strip())
    return ''


class CpuAttributionSampler:
    """Zero-sleep stateful delta sampler for host, LXC, and QEMU VM CPU usage."""

    def __init__(
        self,
        proc_root: str | Path = '/proc',
        cgroup_root: str | Path = '/sys/fs/cgroup',
        qemu_run_dir: str | Path = '/run/qemu-server',
        pve_dir: str | Path = '/etc/pve',
        cpu_count: Optional[int] = None,
        clk_tck: Optional[int] = None,
        noise_floor_pct: float = 1.0,
        per_sample_limit: int = 5,
    ):
        self.proc_root = Path(proc_root)
        self.cgroup_root = Path(cgroup_root)
        self.qemu_run_dir = Path(qemu_run_dir)
        self.pve_dir = Path(pve_dir)
        self._cpu_count = cpu_count if (cpu_count and cpu_count > 0) else (os.cpu_count() or 1)
        if clk_tck and clk_tck > 0:
            self._clk_tck = clk_tck
        else:
            try:
                self._clk_tck = int(os.sysconf('SC_CLK_TCK')) or 100
            except (AttributeError, OSError, ValueError):
                self._clk_tck = 100
        self._noise_floor_pct = max(0.1, float(noise_floor_pct))
        self._per_sample_limit = max(1, int(per_sample_limit))
        self._lock = threading.Lock()
        self._prev: Optional[Tuple[float, Dict[int, Tuple[int, int]], Dict[Tuple[str, int], int]]] = None

    def reset(self) -> None:
        """Clear baseline snapshot (called when host CPU returns below threshold)."""
        with self._lock:
            self._prev = None

    def _snapshot_proc(self) -> Dict[int, Tuple[int, int]]:
        """Read {pid: (starttime, cpu_ticks)} for all numeric entries in proc_root."""
        snap: Dict[int, Tuple[int, int]] = {}
        try:
            entries = os.listdir(self.proc_root)
        except OSError:
            return snap
        for entry in entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            parsed = _parse_proc_stat(_read_bytes(self.proc_root / entry / 'stat'))
            if parsed is not None:
                _comm, starttime, ticks = parsed
                snap[pid] = (starttime, ticks)
        return snap

    def _snapshot_cgroups(self) -> Dict[Tuple[str, int], int]:
        """Read cgroup v2 usage_usec for running LXC containers and QEMU VMs."""
        snap: Dict[Tuple[str, int], int] = {}
        if not (self.cgroup_root / 'cgroup.controllers').is_file():
            return snap

        lxc_root = self.cgroup_root / 'lxc'
        if lxc_root.is_dir():
            try:
                for entry in os.listdir(lxc_root):
                    if not entry.isdigit():
                        continue
                    usage = _parse_cgroup_usage_usec(_read_text(lxc_root / entry / 'cpu.stat'))
                    if usage is not None:
                        snap[('lxc', int(entry))] = usage
            except OSError:
                pass

        qemu_root = self.cgroup_root / 'qemu.slice'
        if qemu_root.is_dir():
            try:
                for entry in os.listdir(qemu_root):
                    match = _QEMU_SCOPE_DIR_RE.match(entry)
                    if not match:
                        continue
                    usage = _parse_cgroup_usage_usec(_read_text(qemu_root / entry / 'cpu.stat'))
                    if usage is not None:
                        snap[('vm', int(match.group(1)))] = usage
            except OSError:
                pass

        return snap

    def _read_qemu_pid_map(self) -> Dict[int, int]:
        """Map live QEMU PIDs to VMIDs using /run/qemu-server/<vmid>.pid."""
        pid_map: Dict[int, int] = {}
        if not self.qemu_run_dir.is_dir():
            return pid_map
        try:
            entries = os.listdir(self.qemu_run_dir)
        except OSError:
            return pid_map
        for entry in entries:
            if not entry.endswith('.pid'):
                continue
            stem = entry[:-4]
            if not stem.isdigit():
                continue
            first_line = _read_text(self.qemu_run_dir / entry).split('\n', 1)[0].strip()
            if first_line.isdigit():
                pid_map[int(first_line)] = int(stem)
        return pid_map

    def _load_guest_names(self) -> Dict[Tuple[str, int], str]:
        """Resolve (kind, vmid) -> guest name from pmxcfs without subprocesses."""
        names: Dict[Tuple[str, int], str] = {}
        if not self.pve_dir.is_dir():
            return names

        # 1. Fast path: .vmlist + .rrd (single pass over pmxcfs status files)
        vmlist_raw = _read_text(self.pve_dir / '.vmlist')
        rrd_raw = _read_text(self.pve_dir / '.rrd')
        if vmlist_raw and rrd_raw:
            try:
                vmlist = json.loads(vmlist_raw)
                ids = vmlist.get('ids', {}) if isinstance(vmlist, dict) else {}
            except (ValueError, TypeError):
                ids = {}
            if isinstance(ids, dict) and ids:
                rrd_names: Dict[str, str] = {}
                for line in rrd_raw.splitlines():
                    parts = line.split(':')
                    if len(parts) < 3:
                        continue
                    key = parts[0]
                    if key.startswith(('pve-vm-9.0/', 'pve2.3-vm/')):
                        vmid_str = key.split('/', 1)[1]
                        guest_name = parts[2]
                        if guest_name and guest_name != 'U':
                            # Prefer pve-vm-9.0 over legacy pve2.3-vm if both exist
                            if vmid_str not in rrd_names or key.startswith('pve-vm-9.0/'):
                                rrd_names[vmid_str] = _sanitize_label(guest_name)
                for vmid_str, entry in ids.items():
                    if not str(vmid_str).isdigit() or not isinstance(entry, dict):
                        continue
                    raw_type = entry.get('type')
                    kind = 'vm' if raw_type == 'qemu' else 'lxc' if raw_type == 'lxc' else None
                    if kind and str(vmid_str) in rrd_names:
                        names[(kind, int(vmid_str))] = rrd_names[str(vmid_str)]

        return names

    def _resolve_guest_name(
        self,
        kind: str,
        vmid: int,
        cached_names: Dict[Tuple[str, int], str],
    ) -> str:
        """Return guest name from cache or fall back to its /etc/pve config file."""
        cached = cached_names.get((kind, vmid))
        if cached:
            return cached
        if not self.pve_dir.is_dir():
            return ''

        folder = 'qemu-server' if kind == 'vm' else 'lxc'
        key_name = 'name' if kind == 'vm' else 'hostname'

        candidates = [self.pve_dir / folder / f'{vmid}.conf']
        members_raw = _read_text(self.pve_dir / '.members')
        if members_raw:
            try:
                nodename = json.loads(members_raw).get('nodename')
                if isinstance(nodename, str) and nodename:
                    candidates.append(self.pve_dir / 'nodes' / nodename / folder / f'{vmid}.conf')
            except (ValueError, TypeError):
                pass

        for conf_path in candidates:
            text = _read_text(conf_path)
            if text:
                parsed = _parse_conf_name(text, key_name)
                if parsed:
                    cached_names[(kind, vmid)] = parsed
                    return parsed
        return ''

    def _read_comm(self, pid: int) -> str:
        comm = _read_text(self.proc_root / str(pid) / 'comm').strip()
        if not comm:
            parsed = _parse_proc_stat(_read_bytes(self.proc_root / str(pid) / 'stat'))
            if parsed is not None:
                comm = parsed[0]
        return _sanitize_label(comm, max_len=32) or f'pid-{pid}'

    def sample(self, now: Optional[float] = None) -> Optional[List[Dict[str, Any]]]:
        """Record a snapshot and return top consumers since the prior snapshot.

        Returns None on the first call (baseline only) or if elapsed time is
        too small to compute a reliable delta.
        """
        ts = time.monotonic() if now is None else float(now)
        proc_snap = self._snapshot_proc()
        cg_snap = self._snapshot_cgroups()

        with self._lock:
            prev = self._prev
            self._prev = (ts, proc_snap, cg_snap)

        if prev is None:
            return None

        prev_ts, prev_proc, prev_cg = prev
        dt = ts - prev_ts
        if dt < 0.1:
            return None

        denom_ticks = dt * self._cpu_count * self._clk_tck
        denom_usec = dt * self._cpu_count * 1_000_000.0
        pid_prefilter = min(0.25, self._noise_floor_pct * 0.5)

        busy_pids: List[Tuple[int, float]] = []
        for pid, (start, ticks) in proc_snap.items():
            old = prev_proc.get(pid)
            if old is None:
                continue
            old_start, old_ticks = old
            if start != old_start:
                continue  # PID was reused between snapshots
            delta = ticks - old_ticks
            if delta <= 0:
                continue
            pct = (delta / denom_ticks) * 100.0
            if pct >= pid_prefilter:
                busy_pids.append((pid, min(100.0, pct)))

        cg_pcts: Dict[Tuple[str, int], float] = {}
        for key, usage in cg_snap.items():
            old_usage = prev_cg.get(key)
            if old_usage is None:
                continue
            delta_u = usage - old_usage
            if delta_u <= 0:
                continue
            pct = min(100.0, (delta_u / denom_usec) * 100.0)
            if pct >= pid_prefilter:
                cg_pcts[key] = pct

        if not busy_pids and not cg_pcts:
            return []

        qemu_pid_map = self._read_qemu_pid_map()
        guest_names = self._load_guest_names()

        vm_pid_totals: Dict[int, float] = defaultdict(float)
        lxc_pid_totals: Dict[int, float] = defaultdict(float)
        lxc_procs: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        host_procs: Dict[str, List[float]] = defaultdict(list)

        for pid, pct in busy_pids:
            cg_text = _read_text(self.proc_root / str(pid) / 'cgroup')
            kind, gid = classify_cgroup(cg_text)
            if kind == 'vm' and gid is not None:
                vm_pid_totals[gid] += pct
                continue
            if kind == 'lxc' and gid is not None:
                lxc_pid_totals[gid] += pct
                comm = self._read_comm(pid)
                lxc_procs[gid][comm].append(pct)
                continue
            comm = self._read_comm(pid)
            # Fall back to /run/qemu-server/<vmid>.pid only when the process
            # actually looks like QEMU/KVM so a stale pidfile after PID reuse
            # can never misattribute an unrelated host process to a VM.
            pidfile_vmid = qemu_pid_map.get(pid)
            if pidfile_vmid is not None and (comm == 'kvm' or comm.startswith('qemu')):
                vm_pid_totals[pidfile_vmid] += pct
            else:
                host_procs[comm].append(pct)

        consumers: List[Dict[str, Any]] = []

        # QEMU VMs: combine cgroup v2 totals (authoritative when present) and PID sums
        all_vmids = set(vm_pid_totals.keys()) | {
            vmid for kind, vmid in cg_pcts if kind == 'vm'
        }
        for vmid in all_vmids:
            cg_val = cg_pcts.get(('vm', vmid), 0.0)
            pid_val = vm_pid_totals.get(vmid, 0.0)
            total_pct = min(100.0, max(cg_val, pid_val))
            if total_pct < self._noise_floor_pct:
                continue
            consumers.append({
                'kind': 'vm',
                'vmid': vmid,
                'name': self._resolve_guest_name('vm', vmid, guest_names),
                'cpu_pct': round(total_pct, 1),
            })

        # LXC containers: guest total + top 1-2 processes inside the container
        all_ctids = set(lxc_pid_totals.keys()) | {
            vmid for kind, vmid in cg_pcts if kind == 'lxc'
        }
        for vmid in all_ctids:
            cg_val = cg_pcts.get(('lxc', vmid), 0.0)
            pid_val = lxc_pid_totals.get(vmid, 0.0)
            total_pct = min(100.0, max(cg_val, pid_val))
            if total_pct < self._noise_floor_pct:
                continue
            inner: List[Dict[str, Any]] = []
            for proc_name, vals in lxc_procs.get(vmid, {}).items():
                proc_pct = min(total_pct, sum(vals))
                if proc_pct >= self._noise_floor_pct:
                    entry: Dict[str, Any] = {
                        'name': proc_name,
                        'cpu_pct': round(proc_pct, 1),
                    }
                    if len(vals) > 1:
                        entry['count'] = len(vals)
                    inner.append(entry)
            inner.sort(key=lambda item: (-item['cpu_pct'], item['name']))
            consumers.append({
                'kind': 'lxc',
                'vmid': vmid,
                'name': self._resolve_guest_name('lxc', vmid, guest_names),
                'cpu_pct': round(total_pct, 1),
                'processes': inner[:2],
            })

        # Host processes grouped by comm
        for proc_name, vals in host_procs.items():
            total_pct = min(100.0, sum(vals))
            if total_pct < self._noise_floor_pct:
                continue
            entry = {
                'kind': 'host',
                'name': proc_name,
                'cpu_pct': round(total_pct, 1),
            }
            if len(vals) > 1:
                entry['count'] = len(vals)
            consumers.append(entry)

        consumers.sort(key=lambda item: (-item['cpu_pct'], item['kind'], str(item.get('vmid', item.get('name', '')))))
        return consumers[:self._per_sample_limit]

    def take_fallback_sample(self, interval: float = 0.5) -> List[Dict[str, Any]]:
        """One-shot short delta sample used only when no high-tick deltas exist."""
        one_shot = CpuAttributionSampler(
            proc_root=self.proc_root,
            cgroup_root=self.cgroup_root,
            qemu_run_dir=self.qemu_run_dir,
            pve_dir=self.pve_dir,
            cpu_count=self._cpu_count,
            clk_tck=self._clk_tck,
            noise_floor_pct=self._noise_floor_pct,
            per_sample_limit=self._per_sample_limit,
        )
        t0 = time.monotonic()
        one_shot.sample(t0)
        time.sleep(max(0.1, float(interval)))
        return one_shot.sample(time.monotonic()) or []


def summarize_consumers(
    samples: Iterable[Optional[List[Dict[str, Any]]]],
    limit: int = 3,
    min_pct: float = 3.0,
) -> List[Dict[str, Any]]:
    """Aggregate per-tick consumer snapshots across a sustained high-CPU window.

    Every non-None delta sample in `samples` counts toward the denominator so a
    brief spike never outranks a workload that sustained the high-CPU episode.
    """
    valid_samples = [s for s in samples if isinstance(s, list)]
    if not valid_samples:
        return []

    sample_count = len(valid_samples)
    totals: Dict[Tuple[str, Any], float] = defaultdict(float)
    meta: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    lxc_proc_totals: Dict[Tuple[str, Any], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    lxc_proc_counts: Dict[Tuple[str, Any], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    host_counts: Dict[Tuple[str, Any], int] = defaultdict(int)

    for sample in valid_samples:
        for item in sample:
            if not isinstance(item, dict):
                continue
            kind = item.get('kind')
            if kind not in ('vm', 'lxc', 'host'):
                continue
            try:
                pct = float(item.get('cpu_pct', 0.0))
            except (TypeError, ValueError):
                continue
            if pct <= 0:
                continue

            if kind in ('vm', 'lxc'):
                vmid = item.get('vmid')
                if vmid is None:
                    continue
                key = (kind, int(vmid))
                totals[key] += pct
                prev_meta = meta.get(key, {'kind': kind, 'vmid': int(vmid), 'name': ''})
                name = _sanitize_label(item.get('name'))
                if name:
                    prev_meta['name'] = name
                meta[key] = prev_meta

                if kind == 'lxc' and isinstance(item.get('processes'), list):
                    for proc in item['processes']:
                        if not isinstance(proc, dict):
                            continue
                        pname = _sanitize_label(proc.get('name'), max_len=32)
                        if not pname:
                            continue
                        try:
                            ppct = float(proc.get('cpu_pct', 0.0))
                        except (TypeError, ValueError):
                            continue
                        if ppct > 0:
                            lxc_proc_totals[key][pname] += ppct
                            pcount = int(proc.get('count', 1) or 1)
                            if pcount > lxc_proc_counts[key][pname]:
                                lxc_proc_counts[key][pname] = pcount
            else:
                name = _sanitize_label(item.get('name'), max_len=32)
                if not name:
                    continue
                key = ('host', name)
                totals[key] += pct
                meta[key] = {'kind': 'host', 'name': name}
                count = int(item.get('count', 1) or 1)
                if count > host_counts[key]:
                    host_counts[key] = count

    ranked: List[Dict[str, Any]] = []
    for key, total_pct in totals.items():
        avg_pct = min(100.0, total_pct / sample_count)
        if avg_pct < min_pct:
            continue
        row = dict(meta[key])
        row['cpu_pct'] = round(avg_pct, 1)
        if row['kind'] == 'lxc':
            procs: List[Dict[str, Any]] = []
            for pname, ptotal in lxc_proc_totals.get(key, {}).items():
                pavg = min(avg_pct, ptotal / sample_count)
                if pavg >= 1.0:
                    pentry: Dict[str, Any] = {'name': pname, 'cpu_pct': round(pavg, 1)}
                    if lxc_proc_counts[key].get(pname, 1) > 1:
                        pentry['count'] = lxc_proc_counts[key][pname]
                    procs.append(pentry)
            procs.sort(key=lambda p: (-p['cpu_pct'], p['name']))
            if procs:
                row['processes'] = procs[:2]
        elif row['kind'] == 'host' and host_counts.get(key, 1) > 1:
            row['count'] = host_counts[key]
        ranked.append(row)

    ranked.sort(key=lambda item: (-item['cpu_pct'], item['kind'], str(item.get('vmid', item.get('name', '')))))
    return ranked[:max(1, int(limit))]
