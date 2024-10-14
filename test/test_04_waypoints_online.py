# This example shows the usage of intermediate waypoints. It will only work with Ruckig Pro or enabled cloud API (e.g. default when installed by pip / PyPI).

from copy import copy, deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Tuple

from ruckig import InputParameter, OutputParameter, Result, Ruckig, Trajectory
import numpy as np

from examples.plotter import Plotter
import plotly.graph_objects as go

@dataclass
class MockMotorConfig:
    velocity_jitter : float = 0.0 # m/s
    acceleration: float | None = None # m/s^2
    time_step: float = 0.01 # seconds
    
@dataclass
class MockMotor:
    mock_motor_config: MockMotorConfig = field(default_factory=MockMotorConfig)
    position: float = field(default=0)
    velocity: float = field(default=0)
    acceleration: float = field(default=0)
    previous_position: float = field(default=0)
    previous_velocity: float = field(default=0)
    previous_acceleration: float = field(default=0)
    position_history: list = field(default_factory=list)
    velocity_history: list = field(default_factory=list)
    acceleration_history: list = field(default_factory=list)
    
    def save_history(self):
        self.previous_position = self.position
        self.previous_velocity = self.velocity
        self.previous_acceleration = self.acceleration
        self.position_history.append(self.position)
        self.velocity_history.append(self.velocity)
        self.acceleration_history.append(self.acceleration)
    
    def process_command(self, target_velocity: float, target_acceleration: float):
        self.save_history()
        
        
        if target_velocity > self.velocity:
            self.acceleration = self.mock_motor_config.acceleration if self.mock_motor_config.acceleration is not None else target_acceleration
        elif target_velocity < self.velocity:
            self.acceleration = -self.mock_motor_config.acceleration if self.mock_motor_config.acceleration is not None else target_acceleration
        else:
            self.acceleration = 0
        
        velocity_jitter = np.random.uniform(-self.mock_motor_config.velocity_jitter, self.mock_motor_config.velocity_jitter)
        additional_velocity = self.acceleration * self.mock_motor_config.time_step + velocity_jitter
        self.velocity = self.velocity + additional_velocity
        # Calculate the current position
        self.position = self.position + (self.previous_velocity * self.mock_motor_config.time_step ) + (0.5 * self.acceleration * self.mock_motor_config.time_step  ** 2)
        pass
        

def generate_offline_trajectory(inp : InputParameter, timestep :float) -> Tuple[list[float], list[float], list[float]]:
    # Calculate entire trajectory at once
    offline_input : InputParameter = InputParameter(1)  # DoFs
    offline_input.current_position = inp.current_position
    offline_input.current_velocity = inp.current_velocity
    offline_input.current_acceleration = inp.current_acceleration
    offline_input.target_position = inp.target_position
    offline_input.target_velocity = inp.target_velocity
    offline_input.target_acceleration = inp.target_acceleration
    offline_input.max_velocity = inp.max_velocity
    offline_input.max_acceleration = inp.max_acceleration
    offline_ruckig = Ruckig(1)  # DoFs, control cycle rate
    offline_trajectory = Trajectory(1)
    offline_result = offline_ruckig.calculate(offline_input, offline_trajectory)
    if offline_result == Result.ErrorInvalidInput:
        raise Exception('Invalid input!')
    offline_positions :list[float]= []
    offline_velocities :list[float]= []
    offline_accelerations :list[float]= []
    for t in np.arange(0, offline_trajectory.duration, timestep):
        pos, vel, acc = offline_trajectory.at_time(t)
        offline_positions.append(pos[0])
        offline_velocities.append(vel[0])
        offline_accelerations.append(acc[0])
        
    return offline_positions, offline_velocities, offline_accelerations

def test_waypoints_online():
    # Create instances: the Ruckig OTG as well as input and output parameters
    mock_motor = MockMotor()
    otg = Ruckig(1, mock_motor.mock_motor_config.time_step)  # DoFs, control cycle rate
    inp : InputParameter = InputParameter(1)  # DoFs
    out = OutputParameter(1, 10)  # DoFs, maximum number of intermediate waypoints for memory allocation
    
    def update_input():
        # the data we read from moteus is the current position, velocity, and acceleration, not the target values
        inp.current_position = [mock_motor.previous_position]
        inp.current_velocity = [mock_motor.previous_velocity]
        inp.current_acceleration = [mock_motor.previous_acceleration]
    update_input()
    
    # Set target
    inp.target_position = [2] # meters
    inp.target_velocity = [0]
    inp.target_acceleration = [0]
    target_position_tolerance = 0.001

    # Set constraints
    inp.max_velocity = [1] # m/s
    inp.max_acceleration = [0.75 ] # m/s^2

    offline_positions, offline_velocities, offline_accelerations = generate_offline_trajectory(inp, mock_motor.mock_motor_config.time_step)

    # Generate the trajectory within the control loop
    out_list = []
    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)
        out_list.append(copy(out))
        
        mock_motor.process_command(out.new_velocity[0], out.new_acceleration[0])
        update_input()
        print(f"Motor Position: {mock_motor.position:.2f} Velocity: {mock_motor.velocity:.2f} Acceleration: {mock_motor.acceleration:.2f}")
        
        # Check if the current position is within the target position tolerance
        position_error = abs(mock_motor.position - inp.target_position[0])
        if position_error <= target_position_tolerance:
            print(f"Position error: {position_error} is within tolerance")
            break  # Exit the loop if within tolerance

    # Plot the trajectory
    
    def plot_mock_motor_history(mock_motor):
        fig = go.Figure()

        time_series = np.arange(0, len(mock_motor.position_history) * mock_motor.mock_motor_config.time_step, mock_motor.mock_motor_config.time_step)
        
        fig.add_trace(go.Scatter(x=time_series, y=mock_motor.position_history, mode='lines+markers', name='Position'))
        fig.add_trace(go.Scatter(x=time_series, y=mock_motor.velocity_history, mode='lines+markers', name='Velocity'))
        fig.add_trace(go.Scatter(x=time_series, y=mock_motor.acceleration_history, mode='lines+markers', name='Acceleration'))
        
        offline_time_series = np.arange(0, len(offline_positions) * mock_motor.mock_motor_config.time_step, mock_motor.mock_motor_config.time_step)
        
        fig.add_trace(go.Scatter(x=offline_time_series, y=offline_positions, mode='lines+markers', name='Offline Position'))
        fig.add_trace(go.Scatter(x=offline_time_series, y=offline_velocities, mode='lines+markers', name='Offline Velocity'))
        fig.add_trace(go.Scatter(x=offline_time_series, y=offline_accelerations, mode='lines+markers', name='Offline Acceleration'))

        fig.update_layout(
            title='Mock Motor History',
            xaxis_title='Time Step',
            yaxis_title='Value',
            legend_title='Legend'
        )

        fig.show()

    plot_mock_motor_history(mock_motor)
    

