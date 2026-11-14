"""vamcore — motor propio de VAM Studio (docs/MIGRACION.md). GPL-3.0.

Importar ``vamcore`` no carga torch; no depende de ASTRA ni de VAMToolbox. Los módulos aceptan arrays de
NumPy y, cuando torch está instalado, también ``torch.Tensor`` (se opera en el dispositivo del tensor).

Convención de ejes (única en todo el paquete):
  volumen     ``[ix, iy, iz]``  — eje de giro en el centro de (ix, iy), z creciente con iz
  sinograma   ``[u, ángulo, z]``

Partes adaptadas de VAMToolbox (GPL-3.0, Regents of the University of California / Taylor lab), con atribución
en la cabecera de cada módulo: ``ops``, ``metrics``, ``filters``, ``dose``.
"""

__version__ = "0.2.0"  # mantener igual que pyproject.toml

from . import config, dose, filters, metrics, ops  # noqa: E402,F401
