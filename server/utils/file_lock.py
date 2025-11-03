"""
Cross-process file locking utilities.
Prevents duplicate jobs across multiple processes/workers.
"""

import errno
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class FileLock:
    """Cross-process file lock with TTL and stale lock cleanup."""
    
    def __init__(self, ttl_seconds: float = 300.0):
        self.ttl = ttl_seconds
    
    def get_lock_path(self, account_id: int, lang: str) -> Path:
        """Get the lock file path for this account/language."""
        base = Path.cwd() / "data" / "gen_locks"
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{int(account_id)}-{str(lang)}.lock"
    
    def acquire(self, account_id: int, lang: str) -> Optional[Path]:
        """Acquire lock, returning Path if successful, None if locked."""
        import time
        lock_path = self.get_lock_path(account_id, lang)
        
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                # Write timestamp to content for debugging
                os.write(fd, str(time.time()).encode("utf-8"))
            finally:
                os.close(fd)
            return lock_path
        except OSError as e:
            if e.errno != errno.EEXIST:
                return None
            
            
            
            # Check if lock is stale and cleanup
            if self._is_stale(lock_path):
                try:
                    if lock_path.exists():
                        lock_path.unlink()
                    
                    # Brief pause to ensure filesystem sync
                    time.sleep(0.1)
                    
                    # Retry once after cleanup
                    retry_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                    try:
                        os.write(retry_fd, str(time.time()).encode("utf-8"))
                    finally:
                        os.close(retry_fd)
                    return lock_path
                except OSError:
                    return None
                except Exception:
                    return None
            
            
            return None
    
    def release(self, lock_path: Optional[Path]) -> None:
        """Release the lock if held."""
        if not lock_path:
            return
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass
    
    def _is_stale(self, lock_path: Path) -> bool:
        """Check if lock is older than TTL."""
        import time
        try:
            stat = lock_path.stat()
            age = (time.time() - stat.st_mtime)
            return age > self.ttl
        except Exception:
            return False
