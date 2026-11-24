"""Unit tests for CPU attribution (host processes, LXC containers, QEMU VMs)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import cpu_attribution  # noqa: E402


def _stat_line(pid: int, comm: str, utime: int, stime: int, starttime: int) -> str:
    """Build a valid Linux /proc/<pid>/stat line with exact field positions."""
    # Fields after (comm): state(3), ppid(4), pgrp(5), session(6), tty_nr(7),
    # tpgid(8), flags(9), minflt(10), cminflt(11), majflt(12), cmajflt(13),
    # utime(14), stime(15), cutime(16), cstime(17), priority(18), nice(19),
    # num_threads(20), itrealvalue(21), starttime(22)
    fields = [
        'R', '1', '1', '1', '0', '-1', '0',
        '0', '0', '0', '0',
        str(utime), str(stime),
        '0', '0', '20', '0', '1', '0',
        str(starttime),
    ]
    return f'{pid} ({comm}) ' + ' '.join(fields) + '\n'


class _SyntheticHost:
    """Fake /proc, /sys/fs/cgroup, /run/qemu-server, and /etc/pve tree."""

    def __init__(self, root: Path, cgroup_v2: bool = True):
        self.root = root
        self.proc = root / 'proc'
        self.cgroup = root / 'cgroup'
        self.qemu_run = root / 'run-qemu'
        self.pve = root / 'pve'
        self.proc.mkdir()
        self.cgroup.mkdir()
        self.qemu_run.mkdir()
        self.pve.mkdir()
        if cgroup_v2:
            (self.cgroup / 'cgroup.controllers').write_text('cpu memory io\n')

    def set_proc(
        self,
        pid: int,
        comm: str,
        utime: int,
        stime: int,
        starttime: int = 1000,
        cgroup: str = '0::/system.slice/test.service\n',
        cmdline_secret: str = 'secret --password=TOPSECRET',
    ):
        pdir = self.proc / str(pid)
        pdir.mkdir(exist_ok=True)
        (pdir / 'stat').write_text(_stat_line(pid, comm, utime, stime, starttime))
        (pdir / 'comm').write_text(comm + '\n')
        (pdir / 'cgroup').write_text(cgroup)
        # Plant a secret in cmdline to guarantee attribution never reads it
        (pdir / 'cmdline').write_bytes(cmdline_secret.encode().replace(b' ', b'\x00'))

    def set_cgroup_usage(self, kind: str, vmid: int, usage_usec: int):
        if kind == 'lxc':
            cdir = self.cgroup / 'lxc' / str(vmid)
        else:
            cdir = self.cgroup / 'qemu.slice' / f'{vmid}.scope'
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / 'cpu.stat').write_text(f'usage_usec {usage_usec}\nuser_usec {usage_usec}\n')

    def set_qemu_pidfile(self, vmid: int, pid: int):
        (self.qemu_run / f'{vmid}.pid').write_text(f'{pid}\n')

    def set_pve_metadata(self, vmlist: dict, rrd_lines: list, configs: dict = None):
        (self.pve / '.vmlist').write_text(json.dumps({'ids': vmlist}))
        (self.pve / '.rrd').write_text('\n'.join(rrd_lines) + '\n')
        for rel, content in (configs or {}).items():
            path = self.pve / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


class CpuAttributionTests(unittest.TestCase):
    def test_classify_cgroup_handles_v2_v1_nested_docker_and_host(self):
        self.assertEqual(
            cpu_attribution.classify_cgroup('0::/lxc/108/ns/system.slice/plex.service\n'),
            ('lxc', 108),
        )
        self.assertEqual(
            cpu_attribution.classify_cgroup(
                '0::/lxc/105/ns/system.slice/docker-abc123def456.scope\n'
            ),
            ('lxc', 105),
        )
        self.assertEqual(
            cpu_attribution.classify_cgroup('5:cpu,cpuacct:/lxc/210/ns/.lxc\n'),
            ('lxc', 210),
        )
        self.assertEqual(
            cpu_attribution.classify_cgroup('0::/qemu.slice/100.scope\n'),
            ('vm', 100),
        )
        self.assertEqual(
            cpu_attribution.classify_cgroup('0::/system.slice/pvedaemon.service\n'),
            ('host', None),
        )

    def test_mixed_vm_lxc_and_host_attribution_with_cgroup_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = _SyntheticHost(Path(tmp), cgroup_v2=True)
            host.set_pve_metadata(
                vmlist={
                    '100': {'node': 'pve1', 'type': 'qemu'},
                    '108': {'node': 'pve1', 'type': 'lxc'},
                },
                rrd_lines=[
                    'pve-vm-9.0/100:1000:ubuntu-srv:running:0:4:0.5',
                    'pve-vm-9.0/108:1000:media-ct:running:0:2:0.5',
                ],
            )
            host.set_qemu_pidfile(100, 2001)

            # 4-core host, clk_tck=100, dt=5s -> 100% host = 2000 ticks = 20_000_000 usec
            # Baseline at t=100.0
            host.set_proc(2001, 'kvm', 1000, 0, cgroup='0::/qemu.slice/100.scope\n')
            host.set_proc(3001, 'ffmpeg', 500, 0, cgroup='0::/lxc/108/ns/system.slice/docker-123.scope\n')
            host.set_proc(4001, 'zstd', 100, 0, cgroup='0::/system.slice/vzdump.service\n')
            host.set_proc(4002, 'zstd', 100, 0, cgroup='0::/system.slice/vzdump.service\n')
            host.set_proc(5001, 'idle-svc', 50, 0)
            host.set_cgroup_usage('vm', 100, 10_000_000)
            host.set_cgroup_usage('lxc', 108, 5_000_000)

            sampler = cpu_attribution.CpuAttributionSampler(
                proc_root=host.proc,
                cgroup_root=host.cgroup,
                qemu_run_dir=host.qemu_run,
                pve_dir=host.pve,
                cpu_count=4,
                clk_tck=100,
            )
            self.assertIsNone(sampler.sample(100.0))

            # Advance t=105.0 (dt=5s):
            # VM 100: +10_000_000 usec = 50.0% host
            # CT 108: +6_000_000 usec = 30.0% host (ffmpeg: +500 ticks = 25.0%, short-lived children = 5%)
            # Host zstd x2: +150 + 150 = 300 ticks = 15.0% host
            # idle-svc: +1 tick = 0.05% (filtered out)
            host.set_proc(2001, 'kvm', 2000, 0, cgroup='0::/qemu.slice/100.scope\n')
            host.set_proc(3001, 'ffmpeg', 1000, 0, cgroup='0::/lxc/108/ns/system.slice/docker-123.scope\n')
            host.set_proc(4001, 'zstd', 250, 0, cgroup='0::/system.slice/vzdump.service\n')
            host.set_proc(4002, 'zstd', 250, 0, cgroup='0::/system.slice/vzdump.service\n')
            host.set_proc(5001, 'idle-svc', 51, 0)
            host.set_cgroup_usage('vm', 100, 20_000_000)
            host.set_cgroup_usage('lxc', 108, 11_000_000)

            result = sampler.sample(105.0)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0], {'kind': 'vm', 'vmid': 100, 'name': 'ubuntu-srv', 'cpu_pct': 50.0})
            self.assertEqual(result[1]['kind'], 'lxc')
            self.assertEqual(result[1]['vmid'], 108)
            self.assertEqual(result[1]['name'], 'media-ct')
            self.assertEqual(result[1]['cpu_pct'], 30.0)
            self.assertEqual(result[1]['processes'], [{'name': 'ffmpeg', 'cpu_pct': 25.0}])
            self.assertEqual(result[2], {'kind': 'host', 'name': 'zstd', 'cpu_pct': 15.0, 'count': 2})
            # Ensure secrets from cmdline never appear anywhere in output
            self.assertNotIn('TOPSECRET', json.dumps(result))

    def test_short_lived_processes_in_lxc_attributed_via_cgroup_v2(self):
        """Fork-heavy build inside CT still attributes CPU to the CT via cpu.stat."""
        with tempfile.TemporaryDirectory() as tmp:
            host = _SyntheticHost(Path(tmp), cgroup_v2=True)
            host.set_pve_metadata(
                vmlist={},
                rrd_lines=[],
                configs={'lxc/115.conf': 'hostname: builder-ct\n[snap1]\nhostname: old-name\n'},
            )
            host.set_cgroup_usage('lxc', 115, 1_000_000)
            sampler = cpu_attribution.CpuAttributionSampler(
                proc_root=host.proc, cgroup_root=host.cgroup,
                qemu_run_dir=host.qemu_run, pve_dir=host.pve,
                cpu_count=2, clk_tck=100,
            )
            sampler.sample(10.0)
            # 2 cores * 5s = 10_000_000 usec = 100%; +8_000_000 = 80%
            host.set_cgroup_usage('lxc', 115, 9_000_000)
            result = sampler.sample(15.0)
            self.assertEqual(result, [{
                'kind': 'lxc', 'vmid': 115, 'name': 'builder-ct',
                'cpu_pct': 80.0, 'processes': [],
            }])

    def test_cgroup_v1_fallback_and_pid_reuse_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = _SyntheticHost(Path(tmp), cgroup_v2=False)
            host.set_pve_metadata(
                vmlist={},
                rrd_lines=[],
                configs={'lxc/202.conf': 'hostname: web-ct\n'},
            )
            host.set_proc(700, 'php-fpm', 100, 0, starttime=500, cgroup='5:cpu,cpuacct:/lxc/202/ns\n')
            host.set_proc(701, 'reused-proc', 100, 0, starttime=500)
            sampler = cpu_attribution.CpuAttributionSampler(
                proc_root=host.proc, cgroup_root=host.cgroup,
                qemu_run_dir=host.qemu_run, pve_dir=host.pve,
                cpu_count=2, clk_tck=100,
            )
            sampler.sample(10.0)
            # dt=5s, 2 cores -> 1000 ticks = 100%
            host.set_proc(700, 'php-fpm', 700, 0, starttime=500, cgroup='5:cpu,cpuacct:/lxc/202/ns\n')
            # PID 701 reused with different starttime -> must be ignored!
            host.set_proc(701, 'new-proc', 900, 0, starttime=9999)
            result = sampler.sample(15.0)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['kind'], 'lxc')
            self.assertEqual(result[0]['vmid'], 202)
            self.assertEqual(result[0]['name'], 'web-ct')
            self.assertEqual(result[0]['cpu_pct'], 60.0)
            self.assertEqual(result[0]['processes'], [{'name': 'php-fpm', 'cpu_pct': 60.0}])

    def test_pbs_host_without_pve_dir_and_parentheses_in_comm(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = _SyntheticHost(Path(tmp), cgroup_v2=False)
            host.pve.rmdir()
            host.qemu_run.rmdir()
            host.set_proc(800, 'worker (io)', 100, 50)
            sampler = cpu_attribution.CpuAttributionSampler(
                proc_root=host.proc, cgroup_root=host.cgroup,
                qemu_run_dir=host.qemu_run, pve_dir=host.pve,
                cpu_count=2, clk_tck=100,
            )
            sampler.sample(10.0)
            host.set_proc(800, 'worker (io)', 500, 150)
            result = sampler.sample(15.0)
            self.assertEqual(result, [{'kind': 'host', 'name': 'worker (io)', 'cpu_pct': 50.0}])

    def test_reset_clears_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = _SyntheticHost(Path(tmp), cgroup_v2=False)
            host.set_proc(900, 'stress', 100, 0)
            sampler = cpu_attribution.CpuAttributionSampler(
                proc_root=host.proc, cgroup_root=host.cgroup,
                qemu_run_dir=host.qemu_run, pve_dir=host.pve,
                cpu_count=1, clk_tck=100,
            )
            self.assertIsNone(sampler.sample(10.0))
            sampler.reset()
            host.set_proc(900, 'stress', 500, 0)
            # After reset, first call only re-establishes baseline
            self.assertIsNone(sampler.sample(15.0))

    def test_stale_qemu_pidfile_does_not_misattribute_host_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = _SyntheticHost(Path(tmp), cgroup_v2=False)
            host.set_qemu_pidfile(100, 999)
            host.set_proc(999, 'vzdump', 100, 0, cgroup='0::/system.slice/vzdump.service\n')
            sampler = cpu_attribution.CpuAttributionSampler(
                proc_root=host.proc, cgroup_root=host.cgroup,
                qemu_run_dir=host.qemu_run, pve_dir=host.pve,
                cpu_count=2, clk_tck=100,
            )
            sampler.sample(10.0)
            host.set_proc(999, 'vzdump', 600, 0, cgroup='0::/system.slice/vzdump.service\n')
            result = sampler.sample(15.0)
            self.assertEqual(result, [{'kind': 'host', 'name': 'vzdump', 'cpu_pct': 50.0}])

    def test_summarize_consumers_averages_across_window_and_filters_noise(self):
        samples = [
            None,  # baseline tick ignored
            [
                {'kind': 'vm', 'vmid': 100, 'name': 'app-vm', 'cpu_pct': 60.0},
                {'kind': 'lxc', 'vmid': 108, 'name': 'plex', 'cpu_pct': 20.0,
                 'processes': [{'name': 'ffmpeg', 'cpu_pct': 18.0}]},
                {'kind': 'host', 'name': 'brief-blip', 'cpu_pct': 6.0},
            ],
            [
                {'kind': 'vm', 'vmid': 100, 'name': 'app-vm', 'cpu_pct': 70.0},
                {'kind': 'lxc', 'vmid': 108, 'name': 'plex', 'cpu_pct': 20.0,
                 'processes': [{'name': 'ffmpeg', 'cpu_pct': 20.0}]},
            ],
            [
                {'kind': 'vm', 'vmid': 100, 'name': 'app-vm', 'cpu_pct': 50.0},
                {'kind': 'lxc', 'vmid': 108, 'name': 'plex', 'cpu_pct': 20.0,
                 'processes': [{'name': 'ffmpeg', 'cpu_pct': 16.0}]},
            ],
        ]
        summary = cpu_attribution.summarize_consumers(samples, limit=3, min_pct=3.0)
        # brief-blip averaged 6/3 = 2.0% < 3.0% so it is excluded
        self.assertEqual(summary, [
            {'kind': 'vm', 'vmid': 100, 'name': 'app-vm', 'cpu_pct': 60.0},
            {'kind': 'lxc', 'vmid': 108, 'name': 'plex', 'cpu_pct': 20.0,
             'processes': [{'name': 'ffmpeg', 'cpu_pct': 18.0}]},
        ])


if __name__ == '__main__':
    unittest.main()
