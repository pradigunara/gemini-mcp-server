"""
Redka Subprocess Manager

Manages redka (Redis-compatible SQLite backend) as a subprocess for
zero-dependency Redis compatibility in uvx deployments.

Uses 'go run' to execute redka directly from GitHub repository,
eliminating binary download and GLIBC compatibility issues.
"""

import atexit
import logging
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Global process handle for cleanup
_redka_process: Optional[subprocess.Popen] = None
_redka_db_path: Optional[str] = None

# Redka repository and version
REDKA_REPO = "github.com/nalgeon/redka/cmd/redka"
REDKA_VERSION = "v0.5.3"


def start_redka_server() -> bool:
    """
    Start redka server as subprocess for Redis compatibility
    
    Returns:
        bool: True if redka started successfully, False otherwise
    """
    global _redka_process, _redka_db_path
    
    if _redka_process and _redka_process.poll() is None:
        logger.debug("Redka server already running")
        return True
    
    try:
        # Check if Go is available
        if not _check_go_available():
            logger.warning("Go not available, falling back to external Redis")
            return False
        
        # Setup database path
        _redka_db_path = _get_redka_db_path()
        
        # Start redka using 'go run' directly from repository
        cmd = [
            "go", "run", 
            f"{REDKA_REPO}@{REDKA_VERSION}",
            "-h", "127.0.0.1",  # Host
            "-p", "6380",       # Port (avoid Redis conflicts)
            _redka_db_path      # Database file path
        ]
        
        logger.info(f"Starting redka server: {' '.join(cmd)}")
        _redka_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
            # Ensure subprocess dies when parent dies  
            start_new_session=True if hasattr(subprocess, 'start_new_session') else False
        )
        
        # Give redka time to start and download dependencies
        time.sleep(3)  # Longer wait for 'go run' which needs to download and compile
        
        # Check if it's still running
        if _redka_process.poll() is None:
            # Set Redis URL to point to redka
            os.environ["REDIS_URL"] = "redis://localhost:6380"
            logger.info("Redka server started successfully on port 6380")
            
            # Register cleanup handlers for various exit scenarios
            atexit.register(stop_redka_server)
            
            # Only register signal handlers if we're the main process
            # This prevents interference with test frameworks and timeout commands
            if __name__ == "__main__" or os.getenv("GEMINI_MCP_MAIN_PROCESS"):
                def signal_handler(signum, frame):
                    logger.info(f"Received signal {signum}, stopping redka server...")
                    stop_redka_server()
                    
                signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
                signal.signal(signal.SIGTERM, signal_handler)  # Termination request
            
            return True
        else:
            stderr = _redka_process.stderr.read().decode() if _redka_process.stderr else "No error output"
            logger.error(f"Redka server failed to start: {stderr}")
            _redka_process = None
            return False
            
    except Exception as e:
        logger.error(f"Failed to start redka server: {e}")
        return False


def stop_redka_server():
    """Stop redka server if running"""
    global _redka_process
    
    if _redka_process and _redka_process.poll() is None:
        try:
            logger.info("Stopping redka server...")
            
            # Try to terminate the process group first (Unix only)
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(_redka_process.pid), signal.SIGTERM)
                else:
                    _redka_process.terminate()
            except (OSError, ProcessLookupError):
                # Process might already be dead, try direct termination
                _redka_process.terminate()
            
            # Wait up to 5 seconds for graceful shutdown
            try:
                _redka_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Redka server didn't stop gracefully, killing...")
                try:
                    if hasattr(os, 'killpg'):
                        os.killpg(os.getpgid(_redka_process.pid), signal.SIGKILL)
                    else:
                        _redka_process.kill()
                except (OSError, ProcessLookupError):
                    pass  # Process already dead
                _redka_process.wait()
            
            logger.info("Redka server stopped")
        except Exception as e:
            logger.error(f"Error stopping redka server: {e}")
        finally:
            _redka_process = None


def is_redka_running() -> bool:
    """Check if redka server is running"""
    return _redka_process is not None and _redka_process.poll() is None


def get_redka_status() -> dict:
    """Get redka server status"""
    return {
        "running": is_redka_running(),
        "pid": _redka_process.pid if _redka_process else None,
        "db_path": _redka_db_path,
        "redis_url": os.getenv("REDIS_URL")
    }


def _check_go_available() -> bool:
    """Check if Go is available on the system"""
    try:
        result = subprocess.run(
            ["go", "version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            logger.debug(f"Go available: {result.stdout.strip()}")
            return True
        else:
            logger.warning("Go command failed")
            return False
    except Exception as e:
        logger.warning(f"Go not available: {e}")
        return False


def _get_redka_db_path() -> str:
    """Get redka database path"""
    # Check environment variable first
    db_path = os.getenv("GEMINI_MCP_DB_PATH")
    if db_path:
        return db_path
    
    # Use home directory if writable
    home_dir = Path.home()
    if home_dir.exists() and os.access(home_dir, os.W_OK):
        return str(home_dir / ".gemini_mcp_redka.db")
    
    # Fallback to temp directory
    return str(Path(tempfile.gettempdir()) / "gemini_mcp_redka.db")