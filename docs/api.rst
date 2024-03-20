API
---

LineRate is designed to be easy to use, while still making the models flexible. To accomplish this,
we have created the ``linerate.model``-module, which defines the abstract ``ThermalModel`` class --
a common interface for thermal rating models, which can compute the heat balance, the thermal
rating and the steady-state conductor temperature.

A ``ThermalModel`` needs to contain all information about the conductor, span and weather. To make
this easy, we have the ``linerate.types``-module, which defines several dataclasses that can keep
track of this information.

While the ``ThermalModel`` interface makes it easy to compute the ampacity of lines, the class
itself does not perform perform any advanced mathematical computations. All physical relationships
are implemented in the ``linerate.equations``-module. The functions in this module represent one
(or sometimes more) function(s) from an academic source (e.g. a technical report) with detailed
information about how the code maps to the equations. The job of the ``ThermalModel`` is,
therefore, to keep track of which functions to call, in which order and what input they require.


.. toctree::
   :caption: Modules
   :maxdepth: 2
   :titlesonly:

   api/model
   api/types
   api/solver
   api/equations/index
