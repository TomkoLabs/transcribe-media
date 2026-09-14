import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class GX10PreflightTests(unittest.TestCase):
    def _preflight(self, directory, version, gpu_name="NVIDIA GB10"):
        root = Path(directory)
        bin_dir = root / "bin"
        bin_dir.mkdir(exist_ok=True)
        for name in ("cmake", "g++", "make", "git", "pkg-config"):
            path = bin_dir / name
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
        gpu_probe = bin_dir / "nvidia-smi"
        gpu_probe.write_text(f"#!/bin/sh\necho '{gpu_name}'\n")
        gpu_probe.chmod(0o755)
        toolkit = root / "cuda"
        if version:
            (toolkit / "bin").mkdir(parents=True, exist_ok=True)
            compiler = toolkit / "bin/nvcc"
            compiler.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release {version}'\n")
            compiler.chmod(0o755)
        return subprocess.run(
            ["bash", "-c", "set -e; source scripts/gx10-runtime.sh; gx10_preflight"],
            env={**os.environ, "TRANSCRIBE_CUDA_ROOT": str(toolkit),
                 "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]},
            text=True, capture_output=True,
        )

    def test_old_cuda_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._preflight(directory, "12.6")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("12.8 or newer", result.stderr)

    def test_blackwell_toolkit_versions_pass_preflight(self):
        for version in ("12.8", "12.9", "13.0"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                result = self._preflight(directory, version)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_compiler_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._preflight(directory, None)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("TRANSCRIBE_CUDA_ROOT", result.stderr)

    def test_other_arm_gpu_does_not_receive_blackwell_only_build(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._preflight(directory, "12.9", "NVIDIA GH200")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("currently targets GB10", result.stderr)

class InstallerFlowTests(unittest.TestCase):
    """Execute the real installer control flow with isolated runtime commands."""

    def _install(self, directory, failure="", cpu=False):
        import shutil
        root = Path(directory)
        (root / "scripts").mkdir()
        shutil.copy("install.sh", root / "install.sh")
        shutil.copy("scripts/gx10-runtime.sh", root / "scripts/gx10-runtime.sh")
        (root / "transcribe_media.py").touch()
        commands = root / "bin"
        commands.mkdir()
        log = root / "commands.log"

        def executable(path, text):
            path.write_text("#!/usr/bin/env bash\nset -eu\n" + text)
            path.chmod(0o755)

        executable(commands / "uname", "echo x86_64\n")
        for name in ("nvidia-smi", "ffmpeg", "ffprobe"):
            executable(commands / name, "exit 0\n")
        executable(root / "transcribe-media", '''
            echo "launcher $*" >> "$INSTALL_TEST_LOG"
            if [[ "$*" == *--prepare-models* && "$INSTALL_TEST_FAILURE" == models ]]; then
                exit 1
            fi
        ''')
        executable(commands / "uv", '''
            echo "uv $*" >> "$INSTALL_TEST_LOG"
            if [[ "$1" == venv ]]; then
                mkdir -p "$ROOT/.venv/bin"
                cp "$ROOT/bin/test-python" "$ROOT/.venv/bin/python"
            fi
            if [[ "$*" == *whl/cu126* && "$INSTALL_TEST_FAILURE" == wheels ]]; then
                exit 1
            fi
        ''')
        executable(commands / "test-python", '''
            cat >/dev/null
            echo "cuda-execution" >> "$INSTALL_TEST_LOG"
            [[ "$INSTALL_TEST_FAILURE" != kernels ]]
        ''')
        # No driver installation, network, real environment replacement or global
        # launcher link. All remaining operations use this temporary checkout.
        script = '''
            source "$1" --skip-system-packages --non-interactive "${@:2}"
            export ROOT
            install_command() { :; }
            main
        '''
        env = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
               "XDG_CONFIG_HOME": str(root / "config"), "HF_TOKEN": "", "HUGGINGFACE_TOKEN": "",
               "INSTALL_TEST_LOG": str(log), "INSTALL_TEST_FAILURE": failure}
        result = subprocess.run(["bash", "-c", script, "installer-test", str(root / "install.sh"),
                                 *(["--cpu"] if cpu else [])], env=env, text=True, capture_output=True)
        return result, log.read_text() if log.exists() else ""

    def test_gpu_install_prepares_default_models_and_requires_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            result, log = self._install(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("launcher --doctor --device cuda", log)
            self.assertIn("launcher --prepare-models --device cuda", log)
            self.assertNotIn("--no-quality", log)
            self.assertIn("Quality mode is enabled by default", result.stdout)
            self.assertIn("Download JSON", result.stdout)
            self.assertIn("speaker-decisions", result.stdout)
            self.assertIn("Copy apply command", result.stdout)

    def test_gpu_failures_never_install_cpu_or_report_success(self):
        for failure in ("wheels", "kernels", "models"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                result, log = self._install(directory, failure)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("whl/cpu", log)
                self.assertNotIn("Installation complete", result.stdout)
                self.assertIn("failed", result.stderr.lower())

    def test_explicit_cpu_install_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            result, log = self._install(directory, cpu=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("whl/cpu", log)
            self.assertNotIn("whl/cu126", log)
            self.assertNotIn("--device cuda", log)
            self.assertIn("launcher --prepare-models", log)
