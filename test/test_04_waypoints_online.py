# This example shows the usage of intermediate waypoints. It will only work with Ruckig Pro or enabled cloud API (e.g. default when installed by pip / PyPI).

from copy import copy
import os
from pathlib import Path

from ruckig import InputParameter, OutputParameter, Result, Ruckig
import numpy as np

from examples.plotter import Plotter


def test_waypoints_online():
    # Create instances: the Ruckig OTG as well as input and output parameters
    otg = Ruckig(1, 0.01, 10)  # DoFs, control cycle rate, maximum number of intermediate waypoints for memory allocation
    inp = InputParameter(1)  # DoFs
    out = OutputParameter(1, 10)  # DoFs, maximum number of intermediate waypoints for memory allocation

    inp.current_position = [0]
    inp.current_velocity = [0]
    inp.current_acceleration = [0]

    inp.intermediate_positions = [
        [1.4],
    ]

    inp.target_position = [2000]
    inp.target_velocity = [0]
    inp.target_acceleration = [0]

    inp.max_velocity = [1000]
    inp.max_acceleration = [750]
    inp.max_jerk = [200]

    inp.interrupt_calculation_duration = 500  # [µs]

    print('\t'.join(['t'] + [str(i) for i in range(otg.degrees_of_freedom)]))

    # Generate the trajectory within the control loop
    out_list = []
    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)

        if out.new_calculation:
            print('Updated the trajectory:')
            print(f'  Calculation duration: {out.calculation_duration:0.1f} [µs]')
            print(f'  Trajectory duration: {out.trajectory.duration:0.4f} [s]')

        print('\t'.join([f'{out.time:0.3f}'] + [f'{p:0.3f}' for p in out.new_position]))
        out_list.append(copy(out))

        out.pass_to_input(inp)

    # Plot the trajectory
    project_path = Path(__file__).parent.parent.absolute()
    file_path = os.path.join(project_path, 'test', 'test_trajectory.pdf')
    
    Plotter.plot_trajectory(file_path, otg, inp, out_list, plot_jerk=True)

