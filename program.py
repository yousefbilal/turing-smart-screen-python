import argparse
import math
import sys
import time
from typing import Optional, Tuple
import re
import serial
from serial.tools import list_ports
import uptime
import psutil
import library.sensors.sensors_librehardwaremonitor as sensors  # type: ignore
import asyncio

cpu_name = None
gpu_name = None


def find_ch340_port() -> Optional[str]:
    """Search available COM ports and return the first CH340 port name (e.g., 'COM5')."""

    try:
        for p in list_ports.comports():
            desc = (p.description or "") + " " + (p.manufacturer or "")
            hwid = p.hwid or ""
            # Common identifiers for CH340: description often contains 'CH340' or 'USB-SERIAL CH340'
            if "CH340" in desc.upper() or "CH340" in hwid.upper():
                return p.device
    except Exception:
        pass
    return None


def open_serial(port: str, baudrate: int) -> serial.Serial:
    return serial.Serial(port=port, baudrate=baudrate, timeout=1)


def fmt(x: float | int | None, precision=1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if isinstance(x, float):
        return f"{round(x, precision)}"
    return str(x)


async def get_disk_speed(interval=1.0):
    start = psutil.disk_io_counters()
    await asyncio.sleep(interval)
    end = psutil.disk_io_counters()

    read_speed = (end.read_bytes - start.read_bytes) / (1024 * 1024 * interval)  # MB/s
    write_speed = (end.write_bytes - start.write_bytes) / (
        1024 * 1024 * interval
    )  # MB/s
    return read_speed, write_speed


async def get_stats(use_gpu: bool) -> Tuple:
    """Returns (cpu_pct, gpu_pct, gpu_mem_used_mb, gpu_temp_c). Some may be NaN."""

    cpu_load = sensors.Cpu.percentage(interval=0.2)
    cpu_freq = sensors.Cpu.frequency() / 1000
    cpu_fans = sensors.Cpu.fan_percent(None)
    cpu_temp = sensors.Cpu.temperature()

    ram_percent = sensors.Memory.virtual_percent()
    ram_used = sensors.Memory.virtual_used() / 1024**3
    ram_free = sensors.Memory.virtual_free() / 1024**3
    ram_total = ram_free + ram_used

    disk = psutil.disk_usage("/")

    disk_total = round(disk.total / 1024**4, 2)
    disk_used = round(disk.used / 1024**4, 2)
    disk_percent = disk.percent

    r_speed, w_speed = await get_disk_speed()

    up_rate, uploaded, dl_rate, downloaded = map(
        lambda x: x / 1024**2, sensors.Net.stats("Wi-Fi", 0)
    )
    gpu_load = math.nan
    gpu_mem_used = math.nan
    gpu_mem_pct = math.nan
    gpu_mem_total = math.nan
    gpu_temp = math.nan

    try:
        if use_gpu:
            gpu_load, gpu_mem_pct, gpu_mem_used, gpu_mem_total, gpu_temp = (
                sensors.Gpu.stats()
            )
    except Exception:
        pass

    t = uptime.uptime()

    minutes = int(t // 60)
    hours, minutes = divmod(minutes, 60)

    return (
        cpu_load,
        cpu_freq,
        cpu_fans,
        cpu_temp,
        f"{hours}:{minutes:02}",
        ram_percent,
        ram_used,
        ram_total,
        disk_total,
        disk_used,
        disk_percent,
        r_speed,
        w_speed,
        gpu_load,
        gpu_mem_pct,
        gpu_mem_used / 1024,
        gpu_mem_total / 1024,
        gpu_temp,
        up_rate,
        uploaded,
        dl_rate,
        downloaded,
    )


def get_cpu_name() -> str:
    for hardware in sensors.handle.Hardware:
        if hardware.HardwareType == sensors.Hardware.HardwareType.Cpu:
            name = hardware.Name

    # Handle AMD Ryzen (e.g., "AMD Ryzen 9 7950X" -> "R9 7950X")
    amd_match = re.match(r".*Ryzen\s*(\d)\s*(\d+\w*).*", name, re.IGNORECASE)
    if amd_match:
        tier = amd_match.group(1)
        suffix = amd_match.group(2)
        return f"R{tier} {suffix}"

    # Handle Intel Core (e.g., "Intel Core i7-13700KF" -> "i7-13700KF")
    intel_match = re.match(r".*Core\s*(i[3579])[- ](\d+\w*).*", name, re.IGNORECASE)
    if intel_match:
        tier = intel_match.group(1).lower()  # e.g., "i7"
        suffix = intel_match.group(2)  # e.g., "13700KF"
        return f"{tier}-{suffix}"

    # Handle Intel Xeon (e.g., "Intel Xeon E5-2678 v3" -> "Xeon E5-2678 v3")
    if "Xeon" in name:
        return name.replace("Intel", "").strip()

    # Handle other CPUs (e.g., Pentium, Celeron)
    if "Pentium" in name or "Celeron" in name:
        return name.replace("Intel", "").strip()

    # Default: return original name if no pattern matches
    return name.strip()


def simplify_gpu_name(gpu_name):
    # Normalize the input (remove extra spaces, make case consistent)
    normalized = re.sub(r"\s+", " ", gpu_name.strip()).upper()

    # Patterns for different GPU vendors
    patterns = [
        # NVIDIA patterns (RTX, GTX, GT, TITAN, etc.)
        (r".*(RTX|GTX|GT|TITAN|GEFORCE)\s*(\d+\s*\w*).*", lambda m: f"{m[1]} {m[2]}"),
        # AMD patterns (RX, Radeon VII, etc.)
        (
            r".*(RX|RADEON\s*VII|PRO\s*W\d+)\s*(\d*\s*\w*).*",
            lambda m: f"{m[1].replace(' ', '')} {m[2]}",
        ),
        # Intel patterns (Arc, Iris Xe, etc.)
        (
            r".*(ARC|IRIS\s*XE|HD\s*GRAPHICS)\s*([A-Z]?\d+\s*\w*).*",
            lambda m: f"{m[1].replace(' ', '')} {m[2]}",
        ),
        # Matrox/other legacy patterns
        (
            r".*(MATROX|QUADRO|TESLA|GRID)\s*([A-Z]?\d+\s*\w*).*",
            lambda m: f"{m[1]} {m[2]}",
        ),
    ]

    for pattern, processor in patterns:
        match = re.match(pattern, normalized)
        if match:
            result = processor(match)
            # Clean up any remaining double spaces
            return re.sub(r"\s+", " ", result).strip()

    # Fallback: Return original name if no pattern matches
    return gpu_name.strip()


def make_csv_line(
    cpu_load,
    cpu_freq,
    cpu_fans,
    cpu_temp,
    uptime,
    ram_percent,
    ram_used,
    ram_total,
    disk_total,
    disk_used,
    disk_percent,
    r_speed,
    w_speed,
    gpu_load,
    gpu_mem_pct,
    gpu_mem_used,
    gpu_mem_total,
    gpu_temp,
    up_rate,
    uploaded,
    dl_rate,
    downloaded,
) -> str:
    return (
        f"CPU_NAME={cpu_name},"
        f"CPU_LOAD={fmt(cpu_load)},"
        f"CPU_FR={fmt(cpu_freq, 2)},"
        f"CPU_FAN={fmt(cpu_fans)},"
        f"CPU_TEMP={fmt(cpu_temp)},"
        f"UPTIME={uptime},"
        f"RAM_PCT={fmt(ram_percent)},"
        f"RAM_U={fmt(ram_used)},"
        f"RAM_T={fmt(ram_total)},"
        f"DISK_T={fmt(disk_total, 2)},"
        f"DISK_U={fmt(disk_used, 2)},"
        f"DISK_PCT={fmt(disk_percent)},"
        f"DISK_R={fmt(r_speed)},"
        f"DISK_W={fmt(w_speed)},"
        f"GPU_NAME={gpu_name},"
        f"GPU_LOAD={fmt(gpu_load)},"
        f"GPU_MEM_PCT={fmt(gpu_mem_pct)},"
        f"GPU_MEM_T={fmt(gpu_mem_total)},"
        f"GPU_MEM_U={fmt(gpu_mem_used)},"
        f"GPU_TEMP={fmt(gpu_temp)},"
        f"UP_RATE={fmt(up_rate)},"
        f"UP={fmt(uploaded)},"
        f"DL_RATE={fmt(dl_rate)},"
        f"DL={fmt(downloaded)}\n"
    )


def log_error_throttled(msg: str, last_ts: float, min_interval: float = 5.0) -> float:
    """Prints msg to stderr if at least min_interval seconds passed since last_ts.

    Returns the (possibly updated) timestamp to store as the new last_ts.
    """
    now = time.monotonic()
    if now - last_ts > min_interval:
        print(msg, file=sys.stderr)
        return now
    return last_ts


def ensure_serial(
    ser: Optional[serial.Serial], port: Optional[str], baud: int, last_err_time: float
) -> tuple[Optional[serial.Serial], Optional[str], float]:
    """Ensure we have an open serial. Auto-detect port if missing and open it.

    Returns (serial_or_none, port_or_none, updated_last_err_time). Sleeps briefly when retrying.
    """
    if ser is not None and ser.is_open:
        return ser, port, last_err_time

    # Detect port if needed
    if not port:
        detected = find_ch340_port()
        if not detected:
            time.sleep(1.0)
            return None, None, last_err_time
        port = detected

    # Try opening
    try:
        ser = open_serial(port, baud)
        time.sleep(0.3)  # allow MCU reset on open
        return ser, port, last_err_time
    except Exception as e:
        last_err_time = log_error_throttled(
            f"Serial open failed on {port!r}: {e}", last_err_time
        )
        time.sleep(1.0)
        return None, port, last_err_time


def try_write(
    ser: serial.Serial, data: str, last_err_time: float
) -> tuple[bool, float]:
    try:
        ser.write(data.encode("utf-8"))
        ser.flush()
        return True, last_err_time
    except Exception as e:
        last_err_time = log_error_throttled(f"Serial write error: {e}", last_err_time)
        try:
            ser.close()
        except Exception:
            pass
        return False, last_err_time


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write CPU/GPU stats to CH340 serial as CSV lines"
    )
    parser.add_argument(
        "--port", help="COM port name (e.g. COM5). If omitted, auto-detect CH340."
    )
    parser.add_argument(
        "--baud", type=int, default=115200, help="Serial baud rate (default: 115200)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    args = parser.parse_args()
    
    asyncio.run(main_async(args))
    
    return 0


    

async def main_async(args) -> int:
    global cpu_name
    global gpu_name
    
    port = args.port or find_ch340_port()
    if not port:
        print("CH340 serial not found. Will keep retrying...", file=sys.stderr)
    
    ser: Optional[serial.Serial] = None
    last_err_time = 0.0  # monotonic seconds
    gpu_enabled = sensors.Gpu.is_available()

    cpu_name = get_cpu_name()
    gpu_name = simplify_gpu_name(sensors.Gpu.gpu_name) if gpu_enabled else None

    try:
        while True:
            ser, port, last_err_time = ensure_serial(
                ser, port, args.baud, last_err_time
            )
            if ser is None:
                continue  # retry detect/open in next iteration
            async with asyncio.TaskGroup() as tg:
                task = tg.create_task(get_stats(gpu_enabled))
                tg.create_task(asyncio.sleep(max(0.05, float(args.interval))))
            
            (
                cpu_load,
                cpu_freq,
                cpu_fans,
                cpu_temp,
                uptime,
                ram_percent,
                ram_used,
                ram_total,
                disk_total,
                disk_used,
                disk_percent,
                r_speed,
                w_speed,
                gpu_load,
                gpu_mem_pct,
                gpu_mem_used,
                gpu_mem_total,
                gpu_temp,
                up_rate,
                uploaded,
                dl_rate,
                downloaded,
            ) = task.result()
            
            line = make_csv_line(
                cpu_load,
                cpu_freq,
                cpu_fans,
                cpu_temp,
                uptime,
                ram_percent,
                ram_used,
                ram_total,
                disk_total,
                disk_used,
                disk_percent,
                r_speed,
                w_speed,
                gpu_load,
                gpu_mem_pct,
                gpu_mem_used,
                gpu_mem_total,
                gpu_temp,
                up_rate,
                uploaded,
                dl_rate,
                downloaded,
            )
            
            print(line)
            ok, last_err_time = try_write(ser, line, last_err_time)
            if not ok:
                ser = None
                continue

    except KeyboardInterrupt:
        pass
    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass



if __name__ == "__main__":
    raise SystemExit(main())
