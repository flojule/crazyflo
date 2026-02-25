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
            'launch/real.launch.xml',
            'launch/real1.launch.xml',
            'launch/sim.launch.xml',
            'launch/launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/config.rviz',
            'config/config1.rviz',
            'config/real.yaml',
            'config/real1.yaml',
            'config/sim.yaml',
            'config/crazyflo.yaml',
        ]),
        ('share/' + package_name + '/data', [
            'data/figure8.csv',
            'data/traj_cf1.csv',
            'data/traj_cf2.csv',
            'data/traj_cf3.csv',
        ]),
        ('share/' + package_name + '/urdf', [
            'urdf/cf2_assembly_with_props.dae',
            'urdf/cf2_assembly.stl',
            'urdf/crazyflie_body.xacro',
            'urdf/crazyflie_description.urdf',
        ]),
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
        ],
        
    },
)
