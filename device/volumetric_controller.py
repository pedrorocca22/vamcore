"""
volumetric_controller.py — Controlador DLP de alta velocidad para impresion volumetrica
========================================================================================
Extiende el DLPC1438 de OpenMLA con metodos optimizados que eliminan los
time.sleep() conservadores y los print() innecesarios.

NO modifica el controlador original de OpenMLA.

Uso:
  from volumetric_controller import FastDMD
  dmd = FastDMD(i2c_bus, spi_bus)
  dmd.init_volumetric_mode(led_pwm=500)
  
  # Ciclo rapido de impresion:
  for img in images:
      dmd.fast_load(img, x_offset, y_offset)
      dmd.fast_swap()
      dmd.fast_expose_ms(150)  # 150ms de exposicion UV
"""

import time
import math
import numpy as np
import RPi.GPIO as GPIO


# Pines del DLPC1438 (mismos que OpenMLA)
PIN_PROJ_ON = 5
PIN_SYS_RDY = 6
PIN_HOST_IRQ = 19
PIN_SPI_RDY = 7
PIN_PRINT_ACTIVE = 13

I2C_ADDR = 0x1B  # Direccion I2C del DLPC1438


class FastDMD:
    """
    Controlador DLP de alta velocidad para impresion volumetrica.
    
    Diferencias con el controlador OpenMLA original:
    - Sin time.sleep() entre comandos I2C (0ms vs 120ms de overhead)
    - Sin print() en el hot loop (evita overhead de consola serial)
    - Metodos fast_* para el ciclo de impresion volumetrica
    - Soporte para exposicion por GPIO (mas preciso que I2C)
    """
    
    SPI_BUFFERSIZE = 65536  # Debe coincidir con /sys/module/spidev/parameters/bufsiz
    
    def __init__(self, i2c_bus, spi_bus):
        self.i2c = i2c_bus
        self.spi = spi_bus
        self.buffer_index = 0
        
        # Configurar pines
        GPIO.setup(PIN_PROJ_ON, GPIO.OUT)
        GPIO.setup(PIN_SYS_RDY, GPIO.IN)
        GPIO.setup(PIN_HOST_IRQ, GPIO.IN)
        GPIO.setup(PIN_SPI_RDY, GPIO.IN)
        GPIO.setup(PIN_PRINT_ACTIVE, GPIO.IN)
        
        # Iniciar DLPC1438
        self._startup()
    
    def _startup(self):
        """Arrancar el DLPC1438 si no esta activo."""
        print("[FastDMD] Iniciando DLPC1438...")
        
        # 1. Comprobar PRIMERO si ya está activo (para evitar reiniciarlo y comerse el pantallazo)
        try:
            self.i2c.read_byte(I2C_ADDR)
            print("[FastDMD] DLPC1438 ya estaba activo")
            self.buffer_index = self._i2c_read(0xC6, 1)[0]
            return
        except IOError:
            pass
            
        # 2. Si no responde, hacemos un ciclo de energía (Hard Reset)
        GPIO.output(PIN_PROJ_ON, GPIO.LOW)
        time.sleep(1)
        
        GPIO.output(PIN_PROJ_ON, GPIO.HIGH)
        t0 = time.time()
        while not GPIO.input(PIN_HOST_IRQ):
            time.sleep(0.01)
        print(f"[FastDMD] HOST_IRQ alto en {time.time()-t0:.2f}s")
        
        time.sleep(1)
        while self.i2c.read_byte(I2C_ADDR) == 0:
            time.sleep(0.5)
        
        self.buffer_index = self._i2c_read(0xC6, 1)[0]
        print(f"[FastDMD] Listo. Buffer activo: {self.buffer_index}")
    
    def _i2c_read(self, reg, length):
        return self.i2c.read_i2c_block_data(I2C_ADDR, reg, length)
    
    def _i2c_write(self, reg, data):
        return self.i2c.write_i2c_block_data(I2C_ADDR, reg, data)
    
    # ================================================================
    # CONFIGURACION (se ejecuta una vez, no necesita ser rapida)
    # ================================================================
    
    def init_volumetric_mode(self, led_pwm=500):
        """
        Configurar el DLPC1438 para modo de impresion volumetrica.
        
        Args:
            led_pwm: Potencia LED UV 0-1023 (0=apagado, 1023=max)
        """
        print(f"[FastDMD] Configurando modo volumetrico (LED PWM={led_pwm}/1023)")
        
        # FPGA control
        self._i2c_write(0xCA, [0b00000000])
        
        # External print config: linear gamma, LED 1
        self._i2c_write(0xA8, [0x00, 0b00000001])
        
        # IMPORTANTE: Configuramos PWM a 0 temporalmente para evitar el "pantallazo" 
        # del splash screen/logo antes de entrar al modo volumétrico.
        pwm_data_zero = [0x00] * 6
        self._i2c_write(0x54, pwm_data_zero)
        
        # Enable FPGA parallel video
        self._i2c_write(0xC3, [0x01])
        
        # Actuator orientation (mismos valores que Elegoo/OpenMLA)
        self._i2c_write(0xC8, [0x03, 0x16, 0x11, 0x06, 0x01])
        
        # Switch to External Print mode
        self._i2c_write(0x05, [0x06])
        time.sleep(0.5)
        
        # Esperar SYS_RDY
        while not GPIO.input(PIN_SYS_RDY):
            time.sleep(0.1)
        
        # Limpiar ambos buffers
        self.clear_buffers()
        
        # AHORA que los buffers están en negro y en modo externo, 
        # restauramos la potencia del LED.
        self.set_led_pwm(led_pwm)
        
        print("[FastDMD] Modo volumetrico activo")
        
    def set_led_pwm(self, led_pwm):
        """Actualizar solo la potencia del LED UV sin reiniciar el DMD."""
        pwm_data = list(led_pwm.to_bytes(2, byteorder='little')) + [0x00]*4
        self._i2c_write(0x54, pwm_data)
    
    def clear_buffers(self):
        """Limpiar ambos buffers FPGA a negro."""
        black = np.zeros((2560, 1440), dtype=np.uint8)
        self._split_spi_send(0, 0, black)
        self.fast_swap()
        self._split_spi_send(0, 0, black)
    
    def shutdown(self):
        """Volver a standby."""
        self._i2c_write(0x05, [0xFF])
        time.sleep(0.5)
        print("[FastDMD] Standby")
    
    # ================================================================
    # METODOS RAPIDOS (hot loop — sin sleep, sin print)
    # ================================================================
    
    def fast_load(self, pixeldata, x_offset, y_offset):
        """
        Enviar imagen a buffer inactivo (SPI). Version rapida sin prints.
        
        Args:
            pixeldata: np.array uint8 2D (ancho, alto)
            x_offset:  offset horizontal en pixeles
            y_offset:  offset vertical en pixeles
        """
        self._split_spi_send(x_offset, y_offset, pixeldata)
    
    def fast_swap(self):
        """Intercambiar buffer activo/inactivo. Sin sleep, sin print."""
        self.buffer_index = not self.buffer_index
        self._i2c_write(0xC5, [self.buffer_index])
    
    def fast_expose_start(self, dark_frames=0):
        """Iniciar exposicion UV indefinida. Sin sleep."""
        self._i2c_write(0xC1, [0x00] 
                        + list(dark_frames.to_bytes(2, byteorder='little'))
                        + [0xFF, 0xFF])
    
    def fast_expose_stop(self):
        """Parar exposicion UV. Sin sleep."""
        self._i2c_write(0xC1, [0x01, 0x00, 0x00, 0x00, 0x00])
    
    def fast_expose_frames(self, num_frames, dark_frames=0):
        """
        Exposicion con duracion controlada por el DLPC1438.
        Duracion = num_frames / 60 Hz.
        Ejemplo: 9 frames = 150ms, 6 frames = 100ms
        """
        self._i2c_write(0xC1, [0x00]
                        + list(dark_frames.to_bytes(2, byteorder='little'))
                        + list(num_frames.to_bytes(2, byteorder='little')))
    
    def fast_expose_ms(self, duration_ms, dark_frames=0):
        """
        Exposicion con duracion en milisegundos.
        Internamente convierte a frames a 60Hz: frames = ceil(ms / 16.67)
        """
        num_frames = max(1, math.ceil(duration_ms / (1000.0 / 60.0)))
        self.fast_expose_frames(num_frames, dark_frames)
    
    def is_exposing(self):
        """Check si PRINT_ACTIVE esta alto (exposicion en curso)."""
        return GPIO.input(PIN_PRINT_ACTIVE)
    
    def wait_exposure_done(self):
        """Esperar a que termine la exposicion (PRINT_ACTIVE baje)."""
        while GPIO.input(PIN_PRINT_ACTIVE):
            pass  # Busy-wait, latencia minima
    
    # ================================================================
    # CICLO DE IMPRESION VOLUMETRICA
    # ================================================================
    
    def volumetric_print_cycle(self, images, x_offset, y_offset, 
                                exposure_ms=150, dark_frames=0,
                                on_frame_exposed=None):
        """
        Ciclo completo de impresion volumetrica optimizado.
        
        Pipeline:
          Frame N en buffer activo, Frame N+1 en buffer inactivo
          1. swap_buffer (N+1 pasa a activo)
          2. expose_start (empieza UV sobre N+1)
          3. Mientras se expone: cargar N+2 en buffer inactivo por SPI
          4. Si on_frame_exposed callback: ejecutar (ej: mover motor)
          5. wait_exposure_done
          6. goto 1
        
        Args:
            images:    lista de np.array uint8 2D (N imagenes)
            x_offset:  offset X en DMD
            y_offset:  offset Y en DMD
            exposure_ms: tiempo de exposicion UV por frame
            dark_frames: frames oscuros entre cambios de buffer
            on_frame_exposed: callback(frame_idx) ejecutado durante exposicion
        
        Returns:
            dict con metricas de timing
        """
        n = len(images)
        frame_times = []
        spi_times = []
        
        # Pre-cargar primera imagen
        self.fast_load(images[0], x_offset, y_offset)
        
        for i in range(n):
            t0 = time.time()
            
            # 1. Swap: la imagen cargada pasa a buffer activo
            self.fast_swap()
            
            # 2. Iniciar exposicion UV  
            if exposure_ms > 0:
                self.fast_expose_ms(exposure_ms, dark_frames)
            
            # 3. Mientras expone, cargar siguiente imagen (pipeline)
            if i + 1 < n:
                t_spi = time.time()
                self.fast_load(images[i + 1], x_offset, y_offset)
                spi_times.append(time.time() - t_spi)
            
            # 4. Callback (mover motor, etc.)
            if on_frame_exposed:
                on_frame_exposed(i)
            
            # 5. Esperar a que termine la exposicion
            if exposure_ms > 0:
                self.wait_exposure_done()
            
            frame_times.append(time.time() - t0)
        
        # Parar UV al final
        self.fast_expose_stop()
        
        return {
            'frame_times': frame_times,
            'spi_times': spi_times,
            'total_time': sum(frame_times),
            'avg_frame_ms': np.mean(frame_times) * 1000,
            'avg_spi_ms': np.mean(spi_times) * 1000 if spi_times else 0,
        }
    
    # ================================================================
    # SPI TRANSFER (basado en OpenMLA, sin prints)
    # ================================================================
    
    def _rowcol_block(self, col_start, col_end, row_start):
        """Formatear bloque de 4 bytes con col/row para SPI."""
        raw = col_start + (col_end << 5) + (row_start << 10) + (0b1111 << 28)
        return list(raw.to_bytes(4, byteorder='little'))
    
    def _split_spi_send(self, offset_w, offset_h, pixel_data):
        """
        Enviar datos de imagen por SPI con subdivision automatica.
        Version sin prints del split_spi_transmission de OpenMLA.
        """
        img_w = pixel_data.shape[0]
        img_h = pixel_data.shape[1]
        
        col_start = math.floor(offset_w / 128)
        row_start = math.floor(offset_h / 2)
        
        pad_w_start = offset_w % 128
        pad_h_start = offset_h % 2
        pad_w_end = (128 - (img_w + pad_w_start) % 128) % 128
        pad_h_end = (2 - pad_h_start + img_h) % 2
        
        padded = np.pad(pixel_data, ((pad_w_start, pad_w_end), (pad_h_start, pad_h_end)))
        
        width = padded.shape[0]
        height = padded.shape[1]
        col_end = int(col_start + width / 128 - 1)
        
        eff_buf = self.SPI_BUFFERSIZE - 10
        num_rows = math.floor((eff_buf / width) / 2) * 2
        num_transfers = math.ceil(height / num_rows)
        
        split_data = np.split(padded, [(i+1)*num_rows for i in range(num_transfers-1)], axis=1)
        
        row_tracker = 0
        for idx, data in enumerate(split_data):
            if idx == 0:
                preamble = np.array(
                    [0x04] + self._rowcol_block(col_start, col_end, row_start) 
                    + [0x00] + list(padded.size.to_bytes(4, byteorder='little')),
                    dtype=np.uint8
                )
            else:
                preamble = np.array(
                    [0x04] + self._rowcol_block(col_start, col_end, row_start + row_tracker) 
                    + [0x00],
                    dtype=np.uint8
                )
            
            row_tracker += int(data.shape[1] / 2)
            data = data.flatten("F")
            
            if idx == num_transfers - 1:
                data = np.append(data, np.array([0x00, 0x00, 0x00, 0x00], dtype=np.uint8))
            
            self.spi.writebytes2(np.insert(data, 0, preamble))
