************
Custom Model
************

The custom model allows you to implement your own operational model entirely in
Python. Instead of choosing one of the built-in models.

However, JuPedSim provides a template class that can be used to start your implemntation. This class is called `CustomOperationalModel`. In it, the methods that need to be implemented by the user are given. In `compute_new_position` the user needs to calculate the next positiion of an agent during a time step. All parameters that are changed during the time step are packed into an update class and the parameters are updated automatically.


Creating a custom model
=======================

To use `CustomOperationalModel` as the base for new operational model, your class needs to subclass or inherit from it. This realized in Python as follows:

.. code:: python

    import jupedsim as jps
    from jupedsim import (
        CustomOperationalModel,
        CustomModelAgentUpdate,
    )

    class MyModel(CustomOperationalModel):

        def __init__(self, param_dict):
            self._dict = param_dict

We created the class `MyModel` and added the constructor function, this is the function that defines fow a model is created and which parameters are used for construction. In this case, we assume that we initiate a `MyModel` instance with a dictionary of values (`param_dict = { "force_constant" : 0.42}`). The values used when creating a model are the same for all agents within the simulation.

We will cover the addition of model parameters that are set per agent later on.

First we will have a look at the other two methods that need to be implemented by the user:

.. code:: python

    import jupedsim as jps
    from jupedsim import (
        CustomOperationalModel,
        CustomModelAgentUpdate,
    )

    class MyModel(CustomOperationalModel):

        def compute_new_position(self, dt, agent, geometry, neighborhood_search):
            # compute new position and return a CustomModelAgentUpdate
            ...
            update = CustomModelAgentUpdate()
            update.position = new_position
            update.velocity = new_velocity
            return update

        def check_model_constraint(self, agent, neighborhood_search, geometry):
            # return True when the agent's state is valid, False otherwise
            return True

`compute_new_position` is the main function of an operational model, based on the current position, velocity and other parameters of an agent, the next position of the agent is computed.
In addition to the position also other parameters that change per iteration step are written into a `CustomModelAgentUpdate` instance. This class holds all values that are updated after this time step. It is also conceptually important to understand that we first compute all new positions of the agents before updating all of them.

`check_model_constraint` is a function that is called when adding new agents to a simulation. Its purpose is to check if no model constraints are violated such as giving negative values to strictly positive model parameters or putting agents to close together.

Setting an instance of `MyClass` as the model for a simulation would be done like so:

.. code:: python

    param_dict = { "force_constant" : 0.42}
    sim = jps.Simulation(model=MyModel(param_dict), geometry=geometry, dt=0.05)



We will introduce the functions and the parameters available in them in detail in the next section.


``compute_new_position``
========================

Called once per agent per time step. Receives:

* **dt** (*float*) -- the simulation time step in seconds.
* **agent** (:class:`~jupedsim.agent.Agent`) -- the agent to update. Provides
  ``position``, ``velocity``, ``orientation``, ``target``, and ``model``
  (a :class:`~jupedsim.models.custom_model.CustomModelAgentParameters`
  instance carrying the per-agent custom parameters).
* **geometry** (:class:`~jupedsim.geometry.Geometry`) -- the walkable area.
  See `Geometry`_ below for full details.
* **neighborhood_search** (:class:`~jupedsim.neighborhood.NeighborhoodSearch`)
  -- spatial query helper. See `NeighborhoodSearch`_ below for full details.

Return a :class:`~jupedsim.models.custom_model.CustomModelAgentUpdate` whose
attributes are applied back to the agent's
:class:`~jupedsim.models.custom_model.CustomModelAgentParameters`. At a
minimum you will usually set ``position`` and ``velocity``.

Geometry
========

:class:`~jupedsim.geometry.Geometry` represents the walkable area and exposes
the walls that bound it.  Inside ``compute_new_position`` it is used to find
walls near the current agent and to compute obstacle forces or boundary
corrections.

Querying walls
--------------

.. code:: python

    walls = geometry.get_walls_in_distance_to(point, distance)

Returns every :class:`~jupedsim.linesegment.LineSegment` whose closest point
to ``point`` is within ``distance`` metres.  Use a conservative distance
(e.g. 3-5 m) so agents react to walls before contact:

.. code:: python

    pos = agent.position
    for wall in geometry.get_walls_in_distance_to(pos, 3.0):
        # wall is a LineSegment
        ...

``linesegments_close_to(point)`` and ``get_walls_close_to(point)`` are
aliases that use the same fixed internal threshold and take no explicit
distance argument.

Boundary and holes
------------------

.. code:: python

    outer = geometry.boundary()  # list of (x, y) tuples -- outer polygon
    inner = geometry.holes()     # list of lists of (x, y) tuples -- holes

These return the raw polygon coordinates of the walkable area.  Useful if
your model needs to reason about the overall shape of the space, but for
per-step wall interaction ``get_walls_in_distance_to`` is the right tool.

Working with LineSegment
------------------------

Each wall returned by the queries above is a
:class:`~jupedsim.linesegment.LineSegment` with the following interface:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Attribute / Method
     - Description
   * - ``wall.p1``
     - First endpoint ``(x, y)``
   * - ``wall.p2``
     - Second endpoint ``(x, y)``
   * - ``wall.closest_point(p)``
     - Point on the segment closest to ``p`` -- the wall contact point for
       repulsion calculations
   * - ``wall.distance_to_point(p)``
     - Shortest distance from ``p`` to the segment

**Computing the outward wall normal** -- the direction a repulsive force should
point -- is a two-step operation:

.. code:: python

    closest = wall.closest_point(agent.position)
    dx = agent.position[0] - closest[0]
    dy = agent.position[1] - closest[1]
    dist = (dx**2 + dy**2) ** 0.5

    if dist > 1e-3:        # avoid division by zero when agent is on the wall
        n_x = dx / dist   # unit normal pointing away from the wall
        n_y = dy / dist

This is the pattern used in ``PythonSocialForceModel._obstacle_force``:

.. code:: python

    def _obstacle_force(self, agent, obstacle):
        closest_point = obstacle.closest_point(agent.position)
        dist = self._distance(agent.position, closest_point)

        if dist < 1e-3:
            return (0.0, 0.0)

        # unit normal away from the wall
        n_x = (agent.position[0] - closest_point[0]) / dist
        n_y = (agent.position[1] - closest_point[1]) / dist

        exp_factor = math.exp(-dist / self.force_distance)
        f_n = self.obstacle_scale * exp_factor

        if dist < self.radius:          # body contact when overlapping
            f_n += self.body_force * (self.radius - dist)

        return (f_n * n_x, f_n * n_y)

NeighborhoodSearch
==================

:class:`~jupedsim.neighborhood.NeighborhoodSearch` provides efficient spatial
queries for finding other agents near the current position.  It wraps a
C++ grid-based data structure that gives O(1) average-case lookup time per
cell, independent of total agent count.

Querying neighbors
------------------

.. code:: python

    neighbors = neighborhood_search.get_neighboring_agents(position, radius)

Returns a list of :class:`~jupedsim.agent.Agent` objects whose position is
within ``radius`` metres of ``position``.  The list is empty when no agents
are found.  Passing a negative radius raises ``ValueError``.

.. code:: python

    pos = agent.position
    for neighbor in neighborhood_search.get_neighboring_agents(pos, 2.0):
        neighbor.position   # (x, y) of the neighbor
        neighbor.model      # their CustomModelAgentParameters
        neighbor.target     # their current waypoint

.. note::
   The returned list **may include the querying agent itself** if its position
   falls within the search radius.  Filter it out by id when needed:

   .. code:: python

       for neighbor in neighborhood_search.get_neighboring_agents(pos, 2.0):
           if neighbor.id == agent.id:
               continue
           # process actual neighbor

Accessing neighbor state
------------------------

Each neighbor carries the same per-agent state as the current agent.  Use
``getattr`` with a default so agents that were not initialised with that field
still work correctly:

.. code:: python

    for neighbor in neighborhood_search.get_neighboring_agents(pos, 2.0):
        n_pos      = neighbor.position
        n_velocity = getattr(neighbor.model, "velocity", (0.0, 0.0))
        n_radius   = getattr(neighbor.model, "radius",   0.3)

This is exactly how ``PythonSocialForceModel.compute_new_position`` collects
the data needed to compute pairwise social forces:

.. code:: python

    for neighbor in neighborhood_search.get_neighboring_agents(pos, 2.0):
        neighbor_pos      = neighbor.position
        neighbor_velocity = getattr(neighbor.model, "velocity", (0.0, 0.0))

        fx, fy = self._social_force(
            pos, velocity, neighbor_pos, neighbor_velocity
        )
        acc_x += fx / self.mass
        acc_y += fy / self.mass

Choosing the search radius
--------------------------

The radius is a trade-off between accuracy and performance:

* **Too small** -- agents outside the radius are ignored entirely, causing
  forces to drop to zero abruptly at the boundary.
* **Too large** -- many agents are examined even though their contribution
  is negligible, slowing down the simulation linearly with agent density.

A practical rule of thumb: set the radius to the distance at which your
interaction term falls below 1 % of its peak value.  For an exponential
repulsion with decay length *B*, that is roughly ``5 * B``.
``PythonSocialForceModel`` uses a 2 m radius because its force distance
is ~0.08 m and the exponential term is effectively zero beyond 2 m.

``check_model_constraint``
==========================

Called once per agent per time step, after ``compute_new_position``. Return
``True`` if the proposed update is valid. Return ``False`` to flag a
constraint violation (the simulation logs it but continues).

Per-agent parameters
====================

:class:`~jupedsim.models.custom_model.CustomModelAgentParameters` is a
flexible container: you can store any attributes you like -- either as a plain
``dict`` or as a dataclass -- and they will be available on the agent's
``model`` property inside ``compute_new_position``.

.. code:: python

    params = jps.CustomModelAgentParameters()
    params.position    = (2.0, 5.0)
    params.journey_id  = journey_id
    params.stage_id    = exit_id
    params.velocity    = (0.0, 0.0)   # custom attribute
    params.desired_speed = 1.34        # custom attribute
    agent_id = sim.add_agent(params)

Inside your model you read them via ``agent.model``:

.. code:: python

    model = agent.model
    speed = getattr(model, "desired_speed", 1.34)

You can also initialise parameters from a dict or dataclass:

.. code:: python

    params = jps.CustomModelAgentParameters(
        {"velocity": (0.0, 0.0), "desired_speed": 1.2}
    )

Example: Python Social Force Model
===================================

A complete reference implementation is included in
``jupedsim.examples.models.pysocial_force``. It re-implements the Helbing &
Molnar (1995) Social Force Model as a custom operational model:

.. code:: python

    import shapely
    import jupedsim as jps
    from jupedsim.examples.models.pysocial_force import PythonSocialForceModel

    geometry = shapely.Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    sim = jps.Simulation(model=PythonSocialForceModel(), geometry=geometry, dt=0.05)

    exit_id   = sim.add_exit_stage([(19, 9), (19, 11), (20, 11), (20, 9)])
    journey_id = sim.add_journey(jps.JourneyDescription([exit_id]))

    params = jps.CustomModelAgentParameters()
    params.position   = (2.0, 10.0)
    params.journey_id = journey_id
    params.stage_id   = exit_id
    params.velocity   = (0.0, 0.0)
    sim.add_agent(params)

    while sim.agent_count() > 0:
        sim.iterate()

.. note::
   Custom models run entirely in Python and are therefore slower than the
   built-in C++ models. They are intended for rapid prototyping and research.
   Once a model is validated, consider porting it to C++ for production use.
