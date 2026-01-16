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
            'config/crazyflies.yaml',
            'config/motion_capture.yaml',
            'config/config.rviz',
        ]),
        ('share/' + package_name + '/data', [
            'data/figure8.csv',
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
            'mission = crazyflo_planner.mission:main',
            'payload = crazyflo_planner.payload:main',
        ],
        
    },
)
