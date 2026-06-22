"""Tegrastats monitoring for Jetson power, temperature, and clock tracking."""
import os
import subprocess
import threading
import time
import re
from typing import Optional, List, Dict, Any
from pathlib import Path


class TegrastatsMonitor:
    """Monitor power, temperature, and clocks via tegrastats."""
    
    def __init__(self, interval_ms: int = 1000, log_file: Optional[str] = None):
        """
        Args:
            interval_ms: Sampling interval in milliseconds.
            log_file: Optional file path to save raw tegrastats output.
        """
        self.interval_ms = interval_ms
        self.log_file = log_file
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.samples: List[Dict[str, Any]] = []
        self.available = self._check_available()
        self.lock = threading.Lock()
    
    def _check_available(self) -> bool:
        """Check if tegrastats command is available."""
        try:
            result = subprocess.run(['which', 'tegrastats'], 
                                    capture_output=True, 
                                    timeout=2)
            return result.returncode == 0
        except Exception:
            return False
    
    def start(self):
        """Start monitoring in background thread."""
        if not self.available:
            print("Warning: tegrastats not available, skipping power/temp monitoring.")
            return
        
        if self.running:
            return
        
        self.running = True
        self.samples = []
        
        # Start tegrastats process
        cmd = ['tegrastats', '--interval', str(self.interval_ms)]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
        except PermissionError:
            print("Warning: tegrastats requires sudo permissions. Trying with sudo...")
            try:
                cmd = ['sudo'] + cmd
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
            except Exception as e:
                print(f"Warning: Failed to start tegrastats with sudo: {e}")
                self.running = False
                return
        except Exception as e:
            print(f"Warning: Failed to start tegrastats: {e}")
            self.running = False
            return
        
        # Start parsing thread
        self.thread = threading.Thread(target=self._parse_loop, daemon=True)
        self.thread.start()
    
    def _parse_loop(self):
        """Parse tegrastats output in background."""
        log_fp = None
        if self.log_file:
            log_fp = open(self.log_file, 'w')
        
        try:
            while self.running and self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if not line:
                    time.sleep(0.01)
                    continue
                
                if log_fp:
                    log_fp.write(line)
                    log_fp.flush()
                
                # Parse the line
                sample = self._parse_line(line)
                if sample:
                    with self.lock:
                        self.samples.append(sample)
        finally:
            if log_fp:
                log_fp.close()
    
    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single tegrastats output line.
        
        Example line format (Jetson Orin Nano):
        01-05-2026 23:46:32 RAM 3534/7620MB (lfb 22x4MB) SWAP 178/12002MB (cached 2MB) 
        CPU [1%@1728,9%@1728,3%@1728,1%@1728,0%@1728,0%@1728] 
        GR3D_FREQ 0% cpu@51.468C soc2@50.281C soc0@50.437C gpu@51.843C tj@51.843C soc1@51.593C 
        VDD_IN 7764mW/7764mW VDD_CPU_GPU_CV 1816mW/1816mW VDD_SOC 2610mW/2610mW
        
        Example line format (JetPack 6.x older format):
        12-25-2024 15:30:45 RAM 2048/8192MB (lfb 512x4MB) SWAP 0/4096MB (cached 0MB) 
        CPU [10%@1900,15%@1900,12%@1900,8%@1900,5%@1900,7%@1900] 
        EMC_FREQ 0%@2133 GR3D_FREQ 0%@1300 VIC_FREQ 114 APE 150 
        POM_5V_IN 2500/2500 POM_5V_GPU 625/625 POM_5V_CPU 312/312
        TEMP CPU 45C GPU 47C SOC 46C
        """
        sample = {
            'timestamp': time.time(),
            'power_w': None,
            'temp_c': None,
            'gpu_clock_mhz': None,
            'cpu_clock_mhz': None,
            'ram_used_mb': None,
            'ram_total_mb': None,
        }
        
        # Parse power - try multiple formats
        # Format 1: VDD_IN (newer format, Orin Nano)
        power_match = re.search(r'VDD_IN\s+(\d+)mW', line)
        if power_match:
            power_mw = float(power_match.group(1))
            sample['power_w'] = power_mw / 1000.0
        else:
            # Format 2: POM_5V_IN (older format)
            power_match = re.search(r'POM_5V_IN\s+(\d+)/(\d+)', line)
            if power_match:
                power_mw = float(power_match.group(1))
                sample['power_w'] = power_mw / 1000.0
        
        # Parse temperature - try multiple formats
        # Format 1: gpu@51.843C (newer format)
        temp_match = re.search(r'gpu@(\d+(?:\.\d+)?)C', line)
        if temp_match:
            sample['temp_c'] = float(temp_match.group(1))
        else:
            # Format 2: GPU 47C (older format)
            temp_match = re.search(r'GPU\s+(\d+(?:\.\d+)?)C', line)
            if temp_match:
                sample['temp_c'] = float(temp_match.group(1))
        
        # Fallback to cpu/soc if gpu temp not found
        if sample['temp_c'] is None:
            # Try cpu@XX.XXC format
            temp_match = re.search(r'cpu@(\d+(?:\.\d+)?)C', line)
            if temp_match:
                sample['temp_c'] = float(temp_match.group(1))
            else:
                # Try CPU XXC format
                temp_match = re.search(r'CPU\s+(\d+(?:\.\d+)?)C', line)
                if temp_match:
                    sample['temp_c'] = float(temp_match.group(1))
                else:
                    # Try soc or tj as last resort
                    temp_match = re.search(r'(?:tj|soc\d*)@(\d+(?:\.\d+)?)C', line)
                    if temp_match:
                        sample['temp_c'] = float(temp_match.group(1))
        
        # Parse GPU clock (GR3D_FREQ)
        gpu_clock_match = re.search(r'GR3D_FREQ\s+\d+%@(\d+)', line)
        if gpu_clock_match:
            sample['gpu_clock_mhz'] = float(gpu_clock_match.group(1))
        
        # Parse CPU clock (from CPU section, average of all cores)
        cpu_clocks = re.findall(r'(\d+)%@(\d+)', line)
        if cpu_clocks:
            clocks = [float(c[1]) for c in cpu_clocks]
            sample['cpu_clock_mhz'] = sum(clocks) / len(clocks) if clocks else None
        
        # Parse RAM usage
        ram_match = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
        if ram_match:
            sample['ram_used_mb'] = float(ram_match.group(1))
            sample['ram_total_mb'] = float(ram_match.group(2))
        
        return sample
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
    
    def get_samples(self) -> List[Dict[str, Any]]:
        """Return collected samples."""
        with self.lock:
            return list(self.samples)
    
    def get_summary(self) -> Dict[str, Any]:
        """Return summary statistics from samples."""
        samples = self.get_samples()
        
        if not samples:
            return {
                'avg_power_w': None,
                'max_temp_c': None,
                'throttling': False,
                'sample_count': 0
            }
        
        powers = [s['power_w'] for s in samples if s['power_w'] is not None]
        temps = [s['temp_c'] for s in samples if s['temp_c'] is not None]
        gpu_clocks = [s['gpu_clock_mhz'] for s in samples if s['gpu_clock_mhz'] is not None]
        
        # Detect throttling: if GPU clock drops below 80% of max for sustained period
        throttling = False
        if gpu_clocks:
            max_clock = max(gpu_clocks)
            threshold = max_clock * 0.8
            
            # Count consecutive samples below threshold
            consecutive_low = 0
            max_consecutive_low = 0
            for clock in gpu_clocks:
                if clock < threshold:
                    consecutive_low += 1
                    max_consecutive_low = max(max_consecutive_low, consecutive_low)
                else:
                    consecutive_low = 0
            
            # If low for >= 30 samples (30 seconds at 1Hz), consider throttled
            if max_consecutive_low >= 30:
                throttling = True
        
        return {
            'avg_power_w': sum(powers) / len(powers) if powers else None,
            'max_temp_c': max(temps) if temps else None,
            'throttling': throttling,
            'sample_count': len(samples),
            'max_gpu_clock_mhz': max(gpu_clocks) if gpu_clocks else None,
            'min_gpu_clock_mhz': min(gpu_clocks) if gpu_clocks else None,
        }
