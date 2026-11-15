"""Jobs: steps over a frozen selection of runs, run as tasks on the hosts of a project.

A job is a frozen request (:func:`darsia.jobs.job.create_job`). Each host runs one
service (:func:`darsia.jobs.service.serve`), which starts ready tasks of any job as
new processes with their job's computing environment (``python -m darsia.jobs``).
A step module declares what a step reads and writes (:mod:`darsia.jobs.step`).

"""
