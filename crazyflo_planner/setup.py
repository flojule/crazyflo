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
            'launch/launch.py',
            'launch/launch.xml']),
        ('share/' + package_name + '/config', [
            'config/crazyflies.yaml',
            'config/motion_capture.yaml',
            'config/config.rviz']),
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
            'hello_world = crazyflo_planner.hello_world:main',
            'arming = crazyflo_planner.arming:main',
            'set_param = crazyflo_planner.set_param:main',
            'figure8 = crazyflo_planner.figure8:main',
            'multi_trajectory = crazyflo_planner.multi_trajectory:main',
        ],
    },
)
