from setuptools import find_packages, setup

package_name = 'crazyflo_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/launch.xml',
            'launch/sim.launch.py',
            'launch/real.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/config.rviz',
            'config/crazyflies.yaml',
            'config/motion_capture.yaml',
            'config/planner_params.yaml',
        ]),
        ('share/' + package_name + '/data', [
            'data/figure8.csv',
            'data/traj_cf1.csv',
            'data/traj_cf2.csv',
            'data/traj_cf3.csv',
        ]),
        # ('share/' + package_name + '/data/multi_trajectory', [
        #     'data/multi_trajectory/traj_1.csv',
        #     'data/multi_trajectory/traj_2.csv',
        # ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='florian-jule',
    maintainer_email='florian.jule@gadz.org',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'payload_sim = crazyflo_planner.payload_sim:main',
            'mission = crazyflo_planner.mission:main',
            'planner = crazyflo_planner.planner:main',
            'figure8 = crazyflo_planner.figure8:main',
        ],
        
    },
)
