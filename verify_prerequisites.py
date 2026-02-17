#!/usr/bin/env python3
"""
Prerequisites Verification Script for AtlasRAG

Checks for:
- Python 3.11+
- Docker installation and daemon status
- Ollama installation

Provides platform-specific installation instructions for any missing prerequisites.
"""

import sys
import subprocess
import platform
import re
from typing import Tuple, List, Dict


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def get_os_type() -> str:
    """Detect operating system type"""
    system = platform.system()
    if system == 'Darwin':
        return 'macos'
    elif system == 'Linux':
        return 'linux'
    elif system == 'Windows':
        return 'windows'
    else:
        return 'unknown'


def check_python_version() -> Tuple[bool, str]:
    """
    Check if Python version is 3.11 or higher

    Returns:
        (is_valid, version_string)
    """
    version_info = sys.version_info
    version_string = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    if version_info.major > 3 or (version_info.major == 3 and version_info.minor >= 11):
        return True, version_string
    else:
        return False, version_string


def check_docker() -> Tuple[bool, str]:
    """
    Check if Docker is installed and daemon is running

    Returns:
        (is_valid, status_message)
    """
    try:
        # Check if docker command exists
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False, "Docker not found"

        docker_version = result.stdout.strip()

        # Check if docker daemon is running
        daemon_check = subprocess.run(
            ['docker', 'info'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if daemon_check.returncode != 0:
            return False, f"{docker_version} (daemon not running)"

        return True, docker_version

    except subprocess.TimeoutExpired:
        return False, "Docker check timed out"
    except FileNotFoundError:
        return False, "Docker not installed"
    except Exception as e:
        return False, f"Error checking Docker: {str(e)}"


def check_ollama() -> Tuple[bool, str]:
    """
    Check if Ollama is installed

    Returns:
        (is_valid, version_string)
    """
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False, "Ollama not found"

        ollama_version = result.stdout.strip()
        return True, ollama_version

    except subprocess.TimeoutExpired:
        return False, "Ollama check timed out"
    except FileNotFoundError:
        return False, "Ollama not installed"
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"


def print_check_result(name: str, is_valid: bool, detail: str) -> None:
    """Print formatted check result"""
    if is_valid:
        status = f"{Colors.GREEN}✓{Colors.END}"
        message = f"{status} {name}: {Colors.GREEN}{detail}{Colors.END}"
    else:
        status = f"{Colors.RED}✗{Colors.END}"
        message = f"{status} {name}: {Colors.RED}{detail}{Colors.END}"

    print(message)


def get_python_installation_instructions(os_type: str) -> str:
    """Get platform-specific Python 3.11+ installation instructions"""

    instructions = {
        'macos': """
  macOS - Using Homebrew:
    brew install python@3.11

  Or download from https://www.python.org/downloads/
""",
        'linux': """
  Linux (Ubuntu/Debian):
    sudo apt update
    sudo apt install python3.11 python3.11-venv python3.11-dev

  Linux (Fedora/RHEL):
    sudo dnf install python3.11 python3.11-devel

  Or download from https://www.python.org/downloads/
""",
        'windows': """
  Windows:
    Download installer from https://www.python.org/downloads/
    Run installer and ensure "Add Python to PATH" is checked

  Or using Chocolatey:
    choco install python311
"""
    }

    return instructions.get(os_type, instructions['linux'])


def get_docker_installation_instructions(os_type: str) -> str:
    """Get platform-specific Docker installation instructions"""

    instructions = {
        'macos': """
  macOS:
    Download Docker Desktop from https://www.docker.com/products/docker-desktop

  Or using Homebrew:
    brew install --cask docker

  Note: After installation, ensure Docker Desktop is running (you'll see the Docker icon in the menu bar)
""",
        'linux': """
  Linux (Ubuntu/Debian):
    sudo apt update
    sudo apt install docker.io docker-compose
    sudo usermod -aG docker $USER
    # Log out and back in for group changes to take effect

  Linux (Fedora):
    sudo dnf install docker docker-compose
    sudo usermod -aG docker $USER

  Start Docker service:
    sudo systemctl start docker
    sudo systemctl enable docker

  Or see official instructions: https://docs.docker.com/engine/install/
""",
        'windows': """
  Windows:
    Download Docker Desktop from https://www.docker.com/products/docker-desktop

  Requirements:
    - Windows 10/11 (Pro, Enterprise, or Education)
    - WSL 2 (Windows Subsystem for Linux 2)
    - Virtualization enabled in BIOS

  See official instructions: https://docs.docker.com/desktop/install/windows-install/
"""
    }

    return instructions.get(os_type, instructions['linux'])


def get_ollama_installation_instructions(os_type: str) -> str:
    """Get platform-specific Ollama installation instructions"""

    instructions = {
        'macos': """
  macOS:
    Download from https://ollama.ai

  Or using Homebrew:
    brew install ollama

  After installation, start Ollama:
    ollama serve

  In another terminal, pull a model:
    ollama pull llama2
""",
        'linux': """
  Linux:
    Download and install from https://ollama.ai/download/linux

  Or run the installer:
    curl https://ollama.ai/install.sh | sh

  Start Ollama service:
    systemctl start ollama

  Pull a model:
    ollama pull llama2
""",
        'windows': """
  Windows:
    Ollama for Windows is in preview
    Download from https://ollama.ai/download/windows

  Or see: https://github.com/ollama/ollama/blob/main/README.md#windows-preview

  After installation, start Ollama and pull a model:
    ollama pull llama2
"""
    }

    return instructions.get(os_type, instructions['linux'])


def print_installation_section(tool_name: str, instructions: str) -> None:
    """Print formatted installation instructions"""
    print(f"\n{Colors.YELLOW}To install {tool_name}:{Colors.END}")
    print(instructions)


def main() -> int:
    """Main verification function"""

    print(f"\n{Colors.BOLD}AtlasRAG Prerequisites Verification{Colors.END}")
    print("=" * 50)

    os_type = get_os_type()
    print(f"Detected OS: {platform.system()} ({os_type})\n")

    # Check prerequisites
    checks = []

    # Python
    python_valid, python_version = check_python_version()
    print_check_result("Python 3.11+", python_valid, python_version)
    checks.append(python_valid)

    # Docker
    docker_valid, docker_detail = check_docker()
    print_check_result("Docker", docker_valid, docker_detail)
    checks.append(docker_valid)

    # Ollama
    ollama_valid, ollama_detail = check_ollama()
    print_check_result("Ollama", ollama_valid, ollama_detail)
    checks.append(ollama_valid)

    # Summary
    passed = sum(checks)
    total = len(checks)

    print("\n" + "=" * 50)
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All prerequisites verified!{Colors.END}")
        print(f"{Colors.GREEN}You're ready to begin Phase 0 setup.{Colors.END}\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ {total - passed} prerequisite(s) missing{Colors.END}")
        print(f"{Colors.YELLOW}Passed: {passed}/{total}{Colors.END}\n")

        # Print installation instructions for missing tools
        if not python_valid:
            print_installation_section("Python 3.11+",
                                      get_python_installation_instructions(os_type))

        if not docker_valid:
            print_installation_section("Docker",
                                      get_docker_installation_instructions(os_type))

        if not ollama_valid:
            print_installation_section("Ollama",
                                      get_ollama_installation_instructions(os_type))

        print(f"\n{Colors.YELLOW}After installing missing tools, run this script again.{Colors.END}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
