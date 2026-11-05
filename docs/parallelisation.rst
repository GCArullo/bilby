================
Parallelisation
================

Many of the samplers supported by Bilby can leverage parallelisation over multiple
CPU cores. There are a range of methods that can be used to achieve this, including
the built-in ``multiprocessing`` module to parallelize over multiple cores on a
single machine, or using MPI to parallelize over multiple nodes on a computing cluster.
This page specifically describes how to use parallelisation of likelihood calls across
multiple processes, rather than parallelisation inside likelihood calls, e.g., using
GPUs via the `Python array API <array_api.rst>`__.

Parallelisation in Bilby relies on global storage of the likelihood and priors inside
``bilby.core.utils.parallel.sampling_convenience_dump``, which must be initialized in
each worker process. We support multiple ways of performing this initialization.

.. note::

    Recent versions of Python include freethreaded builds allowing efficient
    thread-based parallelism. We do not currently support thread-based parallelism
    due to the use of global storage of the likelihood and priors.
    However, we plan to remove this limitation in a future release.

Default parallelisation
=======================

The default approach is to set ``npool`` when calling ``run_sampler``. Bilby
creates a :class:`multiprocessing.Pool`, initializes each worker with the
likelihood and priors, and closes the pool when the run completes. An ``npool``
of ``1`` (the default) or ``None`` runs without a pool.

.. code-block:: python

	import bilby

	result = bilby.run_sampler(
		likelihood=likelihood,
		priors=priors,
		sampler="dynesty",
		npool=4,
	)

The number of workers that a sampler actually uses can depend on that
sampler's options. Refer to the sampler-specific documentation when choosing
the pool size.

``run_sampler`` also passes its pool to the result conversion function. This
means that a conversion function which accepts the optional ``npool`` and
``pool`` arguments can use the same workers after sampling. For example, the
gravitational-wave conversion functions use this to calculate quantities for
each posterior sample in parallel.

.. code-block:: python

	result = bilby.run_sampler(
		likelihood=likelihood,
		priors=priors,
		sampler="dynesty",
		npool=4,
		conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
	)

The conversion function should pass ``npool`` and ``pool`` through to any
Bilby helper which supports parallel processing. A minimal function with the
expected signature is:

.. code-block:: python

	def add_parameters(samples, likelihood=None, priors=None, npool=1, pool=None):
		samples = samples.copy()
		# Calculate and add derived parameters to samples here.
		return samples

Managing a pool with ``bilby_pool``
===================================

Use ``bilby_pool`` when performing parallel work outside ``run_sampler``. It
creates and initializes a pool, yields it, and closes only pools it created.
The context manager also accepts an existing pool without closing it.

.. code-block:: python

	from bilby.core.utils.parallel import bilby_pool

	with bilby_pool(likelihood=likelihood, priors=priors, npool=4) as pool:
		values = list(pool.map(evaluate_sample, samples))

For ``npool=1`` or ``None``, the yielded value is ``None``. Code using the
pool should therefore provide a serial fallback:

.. code-block:: python

	with bilby_pool(likelihood=likelihood, priors=priors, npool=npool) as pool:
		map_function = pool.map if pool is not None else map
		values = list(map_function(evaluate_sample, samples))

Parallel Bilby's MPI pool
=========================

Install the optional dependencies with ``pip install '.[gw,parallel]'`` and
install an MPI runtime providing ``mpiexec``. The ``parallel`` extra pins
``parallel_bilby`` and Schwimmbad to compatible versions.

Select ``pool="parallel_bilby"`` in ``bilby.run_sampler`` and launch the script
with ``mpiexec -n 8 python analysis.py``. Each rank must execute the same call.
The eight ranks comprise one master and seven workers. Bilby initializes the
likelihood and priors on every rank before workers begin waiting, and closes
the pool after sampling and posterior conversion. Cached results are handled
by the master alone.

For Bilby-pipe analyses, use ``bilby_parallel_analysis config.ini`` as the
analysis executable. It launches ``request-cpus`` MPI ranks on the allocated
machine, with at least two ranks required. Keep ``sampler=dynesty``: this
integration uses Parallel Bilby's ``MPIPoolFast`` with Bilby's current Dynesty
driver, checkpoint format, and posterior conversion. Parallel Bilby 2.0.2's
standalone analysis driver uses older Dynesty internals and is not used here.

The real-data and injection launchers under
``examples/gw_examples/data_examples/Cluster_runs_and_utils`` accept
``--parallel-bilby --mpi-cpus 8``. These options select the analysis executable,
request exactly eight CPUs per job, and append ``_parallel_bilby`` to run
labels and output directories. For example:

.. code-block:: bash

    python submit_runs_real_data.py --event GW231123 \
        --likelihood gaussian --num-sine-gaussians 1 \
        --parallel-bilby --mpi-cpus 8 --dry-run --no-container

Remove ``--dry-run --no-container`` for container submissions after publishing
and registering an image for this branch. The container recipe includes MPI
and the optional dependencies. Build it with
``make publish BILBY_BRANCH=codex/parallel-bilby BILBY_PIPE_BRANCH=sine_gaussians_addition``
from ``container_creation`` after pushing the Bilby branch. This is a
single-machine MPI launch; it does not request multiple cluster nodes.

Using a custom pool
===================

To run with a custom pool, e.g., using MPI, create a pool such as
:class:`schwimmbad.MPIPool` and pass it to ``run_sampler``. A user-supplied pool
remains the caller's responsibility to close. Each MPI rank must initialize
Bilby's worker globals before worker ranks begin waiting for work.
Run the script with an MPI launcher, for example ``mpiexec -n 4 python analysis.py``.

.. code-block:: python

	import sys

	import bilby
	from schwimmbad import MPIPool
	from bilby.core.utils.parallel import initialize_global_variables

	pool = MPIPool(use_dill=True)
	initialize_global_variables(
		likelihood=likelihood,
		priors=priors,
		search_parameter_keys=["mass_1", "mass_2"],
		use_ratio=False,
		parameters=priors.sample(),
	)

	if not pool.is_master():
		pool.wait()
		sys.exit(0)

	try:
		result = bilby.run_sampler(
			likelihood=likelihood,
			priors=priors,
			sampler="dynesty",
			pool=pool,
		)
	finally:
		pool.close()

Set ``search_parameter_keys`` to the names of the parameters sampled by the
run, and set ``use_ratio`` to match the requested likelihood evaluation. For
more control, the same initialization pattern applies to any pool-like object
that supports ``map`` and is supplied through the ``pool`` argument.
