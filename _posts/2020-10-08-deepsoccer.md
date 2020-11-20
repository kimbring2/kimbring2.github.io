# Introduction
There are currently various attempts to apply Deep Reinforcement to actual robot. The characteristics of Reinforcment Learning require over hundred of the initialization of the environment and many data for Deep Learning making it impossible to train the robot in the actual environment.

To overcome these difficulties, people first create a simulation that looks exactly like the real world. In this simulation, training is advanced for the virtual robot. Since then, many researches try to use the tranined model for an actual robot.

In this post, I'll try to confirm these methods are actually working. The test uses a robot called Jetbot that has a GPU on its main board. And the environment is a soccer that is familiar to many people.

Code for that post can be found on the [DeepSoccer Github](https://github.com/kimbring2/DeepSoccer).

All the products used in the tests are inexpensive and can be purchased in Amazon. Thus, you can easily try to reproduce them.

# Table of Contents
1. [Soccer robot design](#soccer_robot_design)
    1. [Soccer robot design(Wheel)](#soccer_robot_design_wheel)
    2. [Soccer robot design(Solenoid)](#soccer_robot_design_solenoid)
    3. [Soccer robot design(Lidar)](#soccer_robot_design_lidar)
    4. [Soccer robot design(Infrared)](#soccer_robot_design_infrared)
    5. [Soccer robot design(Integration)](#soccer_robot_design_integration)
    6. [Soccer robot design(Teleoperation)](#soccer_robot_design_teleoperation)
2. [Reinforcement Learning test](#rl_test)
    1. [Network structure](#rl_network_structure)
    2. [Reward setting](#rl_reward_setting)
    3. [Training result](#rl_training_result)
    4. [Changing reward setting for accurate control](#rl_change_reward_setting)
    5. [Training result after revision](#rl_second_training_result)
3. [Real Jetbot test](#real_jetbot_test)
    1. [Controlling original Jetbot](#control_original_jetbot)
    2. [Deep Learning model for Jetbot](#deep_learning_model_jetbot)
    3. [Transfering trained Tensorflow weight to Jetbot](#transfer_tensorflow_model)

<a name="soccer_robot_design"></a>
# Soccer robot design
I conclude that no matter how much football fields I changed, Jetbot need to take a soccer ball or kick it. Thus, I decided to design a robot exclusively for soccer and using the Jetson Nano.

Fortunately, 3d model of NVIDIA Kaya robots and robot participating in Robocup are available online. By using both of these as a reference and utilizing 3d printer, I can easily create a Jetbot for football. And recently, the price of 3D printers has dropped sharply. Thus, I manage to find one printer to create a soccer robot at a very affordable price.

1. [Kaya robot model](https://cad.onshape.com/documents/03aa2560e7a40b2b7da40e12/w/001dbb6db63b0092c9ea5823/e/37043abce9062fab02c40889)
2. [Robocup robot model](https://www.stlfinder.com/model/naghshe-jahan-2010-robocup-soccer-robot/2284603/)
3. [3D printer](https://ko.aliexpress.com/item/32829861835.html?spm=a2g0s.9042311.0.0.42964c4dsLAMTk)

Jetbot's soccer robot design has two primary focuses. Kaya robot's motors, screws, batteries, electronics, etc., initially offered by NVIDIA, are the latest parts sold today and should be used wherever possible. Second, the overall design of the robot should be as similar as possible to the design of the robot that entered Robotbup.

To achieve such a goal, two robot parts are once separated and measured to confirm what parts are needed to be newly designed or modified.

The new robot will have a solenoid in the center for a kicking and the required parts will be produced using a 3D printer.

<a name="soccer_robot_design_wheel"></a>
## Soccer robot design(Wheel)
The power supply to the dynamixel is 12V, which utilizes the Jetbot of WaveShares main board which has 3 18560 battery.

<img src="/assets/circuit_design_soccer.png" width="800">

The U2D2 board and U2D2 power hub can be purchased at the Robotis shopping mall. However, if you have existing one, you can change only the 12V power supply method and use the rest as it is.

1. [U2D2](http://www.robotis.us/u2d2/)
2. [U2D2 Power Hub](http://www.robotis.us/u2d2-power-hub-board-set/)

It is judged that the size omniwheel is too large. Thus, I decided to replace it with a slightly smaller wheel. Basically, onmiwheel product which provides a 3D model is selected for using in Gazebo simualtion.

1. [Omniwheel shop(Korea local shop)](http://robomecha.co.kr/product/detail.html?product_no=10&cate_no=1&display_group=2)
2. [Omniwheel 3D model](https://cad.onshape.com/documents/9a91ce8d931df48891a33741/w/d07aae74b658bfdb32b3c1a2/e/55f0a4a11d07b8ae71bce952)

[![Dynamixel test 1](https://img.youtube.com/vi/4q6_ML3Ii8o/0.jpg)](https://youtu.be/4q6_ML3Ii8o "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

ID and communication method and firmware version of of Dynamixel can given via a program provided by ROBOTIS. I test a motor seperately before installing it to robot body. A power supply and TTL communication can be done by using a U2D2 board and power hub.

<img src="/assets/assembly_1.jpg" height="400"> <img src="/assets/assembly_2.jpg" height="400">

After confirming only the operation of the motor separately, the motor and the control, power board are combined to the robot body. 

<img src="/assets/ccw_setting.png" width="400"> <img src="/assets/speed_setting.png" width="400">

In the case of dynamixel, the initial mode is the joint mode. Mode is needed to be changed to the wheel mode for using soccer robot. This can be achieved by setting the CCW Angle Limit of motor to 0. To rotate the motor set in the wheel mode at a specific speed, you just need to give a specific value to Moving Speed.

[![Dynamixel test 2](https://img.youtube.com/vi/3NiTv4gDRWQ/hqdefault.jpg)](https://youtu.be/3NiTv4gDRWQ "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

Next, I test adjusting the speed of dynamixel using rostopic, as in the previous Jetbot.

[![Dynamixel test 3](https://img.youtube.com/vi/VT6AOI11sbs/hqdefault.jpg)](https://youtu.be/VT6AOI11sbs "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

The Jetbot Soccer version uses an omniwheel that has a many sub wheel. In order to properly simulate this with Gazebo, we must make sure that each sub wheel rotates correctly. First, I check it using RViz in the same way as a main wheel.

[![Omniwheel RVIz test](https://img.youtube.com/vi/Oa-rRioxU7M/hqdefault.jpg)](https://youtu.be/Oa-rRioxU7M "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

After completing the test with RVIz, the test is similarly performed with Gazebo. It is confirmed that when the friction with the floor is large, the phenomenon that the sub wheel do not rotate properly is occurred. Finding the optimal friction parameters will be an important task.

[![Omniwheel Gazebo test](https://img.youtube.com/vi/0bvrdl4Z4Lo/hqdefault.jpg)](https://youtu.be/0bvrdl4Z4Lo "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

<a name="soccer_robot_design_solenoid"></a>
## Soccer robot design(Solenoid)
The mechanism for controlling the ball is composed of a rubber roller for fixing and a solenoid electromagnet for kicking.

1. [Engraving rubber roller(Made in Korea)](
https://www.ebay.com/itm/50mm-Engraving-Rubber-Roller-Brayer-Stamping-Printing-Screening-Tool-Korea-/153413802463)

For the part for grabiing the soccer ball, I use a part of engraving roller. The core of the roller is made by 3D printer and connected to a DC motor which is included in origin Jetbot.

[![Roller test 1](https://img.youtube.com/vi/u3jQLBPoC7Q/0.jpg)](https://youtu.be/u3jQLBPoC7Q "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

After completing the operation test with the roller alone, it is mounted on the of the robot second body layer. Since then, I test whether robot can actually hold the ball.

[![Roller test 2](https://img.youtube.com/vi/ve6R_gJHDgg/0.jpg)](https://youtu.be/ve6R_gJHDgg "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

For the part for kicking the soccer ball, I use a solenoid electromagnet. To control the solenoid electromagnet in the ROS, we should control GPIO using code. I use a GPIO library of Jetson (https://github.com/NVIDIA/jetson-gpio) provided by NVIDIA.

It is determined that directly connecting the solenoid motor directly to the 12V power supply can not enough force to kick the ball far. Thus, large capacity capacitor and charging circuit for it is added. Thankfully I could find circuit and component for this at https://drive.google.com/file/d/17twkN9F0Dghrc06b_U9Iviq9Si9rsyOG/view.

<img src="/assets/reinforced_solenoid.png" width="800">

1. [Capacitor buying link](https://www.aliexpress.com/item/32866139188.html?spm=a2g0s.9042311.0.0.2db94c4dNsaPDZ)
2. [Capacitor Charger](https://www.aliexpress.com/item/32904490215.html?spm=a2g0s.9042311.0.0.27424c4dANjLyy)
3. [Limit switch(Relay module can replace it)](https://www.aliexpress.com/item/32860423798.html?spm=a2g0s.9042311.0.0.2db94c4dNsaPDZ)

After a 250v 1000uf capacitor and a ±45V-390V capacitor charger are added, a solenoid can push a heavy a billiard ball to considerable distance.

[![Solenoid test 5](https://img.youtube.com/vi/n_TAL5K73aA/hqdefault.jpg)](https://youtu.be/n_TAL5K73aA "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

Because the wheel and roller part have all simple structure, they can be implemented using the default controller of Gazebo. However. the solenoid part need a custom controller since it has spring part. You can find a official tutorial for that at http://gazebosim.org/tutorials?tut=guided_i5&cat=. 

<img src="/image/Spring-Constant.jpg" width="600">

```
namespace gazebo
{
	class SolenoidElectromagnetSpringPlugin : public ModelPlugin
	{
		public: SolenoidElectromagnetSpringPlugin() {}     
...
...
...

public: void OnRosMsg(const std_msgs::Float32ConstPtr &_msg)
{
	std::cout << "_msg->data: " << _msg->data << std::endl;
		 
	for (int i = 0; i < 10; i++)
	    this->joint->SetForce(0, 1000 * float(_msg->data));
}

protected: void OnUpdate()
{
	double current_angle = this->joint->Position(0);
	for (int i = 0; i < 5; i++)
		this->joint->SetForce(0, (this->kx * (this->setPoint - current_angle)));
}
```

After testing the tutorial plugin first, add OnRosMsg, OnUpdate function to ModelPlugin. The OnRosMsg function is for receiving message transmitted to rostopic which include force applied to the solenoid by battery. The OnUpdate function is a part for defining a force applied by the spring.

```
<gazebo>
  <plugin name="stick_solenoid_electromagnet_joint_spring" filename="libsolenoid_electromagnet_joint_spring_plugin.so">
    <kx>1000</kx>
    <set_point>0.01</set_point>
    <joint>stick</joint>
  </plugin>
</gazebo>
```

Adding the above part to jetbot_soccer.gazebo after defining the plugin can make it possible to use a custom pluging in that joint.

<a name="soccer_robot_design_lidar"></a>
## Soccer robot design(Lidar)
Soccer robot need to check a obstacle of front side. Using only camera sensor is not enough for that. Thus, I decide adding lidar sensor.

<img src="/assets/lidar_circuit.png" width="800">

The lidar sensor I use can measure only on the front side. It requires 5V, GND and UART GPIO pins of Jetson Nano.

[![lidar test 1](https://img.youtube.com/vi/2b6BUH5tF1g/sddefault.jpg)](https://youtu.be/2b6BUH5tF1g "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

After checking operation of lidar sensor at simulation. I also check real lidar sensor can measure distance of front obstacle.  

[![lidar test 2](https://img.youtube.com/vi/LNNTYWV4W0Y/sddefault.jpg)](https://youtu.be/LNNTYWV4W0Y "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>


<a name="soccer_robot_design_infrared"></a>
## Soccer robot design(Infrared)
Robot need know it is holding holding ball now because camera can not see a lower part of robot. In order to solve these problems, I decide to add an infrared obstacle detection sensor at side of the roller that holds the ball.

<img src="/assets/infrared_circuit.png" width="450">

The infrared sensor uses GPOI as an input directuib, as opposed to the solenoid, which used GPIO as an output. If an obstacle is detected, the sensor gives a different signal to Jetson Nano.

[![Infrared test](https://img.youtube.com/vi/ilSl8ReIZsA/hqdefault.jpg)](https://youtu.be/ilSl8ReIZsA "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

After mounting the infrared sensor in real robot, I also confirm that the same operation can be performed in infrared sensor of Gazebo simulation.

<a name="soccer_robot_design_teleoperation"></a>
## Soccer robot design(Teleoperation)
It makes sense that the default motions performed in Gazebo simulations can be done in a real robot also. In order to confirm tat, the motion control function of  original version of Jetbot using the gamepad is slightly modified for Jetbot soccer version.

<img src="/assets/teleoperation_1.png" width="600">

You probably remember the gamepad UI that is in orginal teleoperation.ipynb file. Only 0, 1 scroolbal is used for controlling the left and right wheel in that file. In my case, I set it as above for controlling roller, stick, omniwheel.

```
from dynamixel_sdk import *                    # Uses Dynamixel SDK library

def omniwheel_move(motor_id, speed):
    # Control table address
    ADDR_MX_TORQUE_ENABLE      = 24               # Control table address is different in Dynamixel model
    ADDR_MX_GOAL_POSITION      = 30
    ADDR_MX_PRESENT_POSITION   = 36
    ADDR_MX_MOVING_SPEED       = 32

    # Protocol version
    PROTOCOL_VERSION            = 1.0               # See which protocol version is used in the Dynamixel
    
    # Default setting
    DXL_ID = motor_id
    #DXL_ID                      = 1                # Dynamixel ID : 1
    BAUDRATE                    = 57600             # Dynamixel default baudrate : 57600
    DEVICENAME                  = '/dev/ttyUSB0'    # Check which port is being used on your controller
                                                    # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

    TORQUE_ENABLE               = 1                 # Value for enabling the torque
    TORQUE_DISABLE              = 0                 # Value for disabling the torque
    DXL_MINIMUM_POSITION_VALUE  = 10           # Dynamixel will rotate between this value
    DXL_MAXIMUM_POSITION_VALUE  = 4000            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
    DXL_MOVING_STATUS_THRESHOLD = 20                # Dynamixel moving status threshold

    index = 0
    dxl_goal_position = [DXL_MINIMUM_POSITION_VALUE, DXL_MAXIMUM_POSITION_VALUE]         # Goal position

    # Initialize PortHandler instance
    # Set the port path
    # Get methods and members of PortHandlerLinux or PortHandlerWindows
    portHandler = PortHandler(DEVICENAME)

    # Initialize PacketHandler instance
    # Set the protocol version
    # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open port
    if portHandler.openPort():
        #print("Succeeded to open the port")
        pass
    else:
        print("Failed to open the port")
        print("Press any key to terminate...")
        getch()
        quit()

    # Set port baudrate
    if portHandler.setBaudRate(BAUDRATE):
        #print("Succeeded to change the baudrate")
        pass
    else:
        print("Failed to change the baudrate")
        print("Press any key to terminate...")
        getch()
        quit()

    # Enable Dynamixel Torque
    #dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, speed)
    #print("dxl_comm_result: " + str(dxl_comm_result))
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        pass
        #print("Dynamixel has been successfully connected")
```
```
# Stop
def stop():
    omniwheel_move(1,0)
    omniwheel_move(2,0)
    omniwheel_move(3,0)
    omniwheel_move(4,0)
```
```
import traitlets
from traitlets.config.configurable import Configurable

class OmniWheelStop(Configurable):
    value = traitlets.Float()

    def __init__(self, *args, **kwargs):
        super(OmniWheelStop, self).__init__(*args, **kwargs)  # initializes traitlets
    
    @traitlets.observe('value')
    def _observe_value(self, change):
        self._write_value(change['new'])

    def _write_value(self, value):
        """Sets motor value between [-1, 1]"""
        stop()

    def _release(self):
        """Stops motor by releasing control"""
        print("_release")
```

```
import RPi.GPIO as GPIO
import time

def solenoid_active():
    # Pin Definitions
    output_pin = 18  # BCM pin 18, BOARD pin 12
    
    # Pin Setup:
    GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

    print("Starting demo now! Press CTRL+C to exit")
    curr_value = GPIO.HIGH
    time.sleep(1)
    
    print("Outputting {} to pin {}".format(curr_value, output_pin))
    GPIO.output(output_pin, curr_value)
    #curr_value ^= GPIO.HIGH
    GPIO.cleanup()
    
solenoid_active()

class SolenoidKick(Configurable):
    value = traitlets.Float()

    def __init__(self, *args, **kwargs):
        super(SolenoidKick, self).__init__(*args, **kwargs)  # initializes traitlets
    
    @traitlets.observe('value')
    def _observe_value(self, change):
        self._write_value(change['new'])

    def _write_value(self, value):
        """Sets motor value between [-1, 1]"""
        solenoid_active()

    def _release(self):
        """Stops motor by releasing control"""
        print("_release")
	

```

```
from jetbot import Robot
import traitlets

robot = Robot()

solenoid_link = traitlets.dlink((controller.buttons[4], 'value'), (solenoid_kick, 'value'), transform=lambda x: -x)

roller_link = traitlets.dlink((controller.axes[1], 'value'), (robot.right_motor, 'value'), transform=lambda x: -x)

stop_link = traitlets.dlink((controller.buttons[5], 'value'), (omniwheel_stop, 'value'), transform=lambda x: -x)
forward_link = traitlets.dlink((controller.buttons[0], 'value'), (omniwheel_forward, 'value'), transform=lambda x: -x)
left_link = traitlets.dlink((controller.buttons[1], 'value'), (omniwheel_left, 'value'), transform=lambda x: -x)
right_link = traitlets.dlink((controller.buttons[2], 'value'), (omniwheel_right, 'value'), transform=lambda x: -x)
back_link = traitlets.dlink((controller.buttons[3], 'value'), (omniwheel_back, 'value'), transform=lambda x: -x)
```

Like the original method, using the gamepad requires combining Traitlets package of python and control code of hardware. Above code is a sample for I use for omniwheel control. First, Dynamixel SDK is used for creating function that gives direction and speed to each omniwheel, and then use these function to create function for the robot to move forward, turn left, turn right, reverse, and stop. Then, using this function, a configurable class of traitlets is created, and the class declared at the end is connected to the gamepad UI using the dlink function of Traitlets.

[![Teleoperation test 1](https://img.youtube.com/vi/vONoIruznlw/hqdefault.jpg)](https://www.youtube.com/watch?v=vONoIruznlw "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

As you can see in the video, you can see the basic movement and ball dribbling are working well. However, kicking power is weak. It is considered that  voltage , current are insufficient because battery and solenoid are directly connected. To solve these problem, I decide to use the method of collecting the high voltage using a large capacitor and then releasing it to the solenoid at once.

[![Teleoperation test 2](https://img.youtube.com/vi/xpfglVhzQOg/hqdefault.jpg)](https://youtu.be/xpfglVhzQOg "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

I test again basic soccer function after adding a large capacitor. It seems like that the kick function is reinforced and it can kick a balls to a long distance than before.

<a name="rl_test"></a>
# Reinforcement Learning test
For this project, I chose Deep Reinforcement Learning as the control method for Jetbot playing soccer. This method train robot control neural networks via reward.

<a name="rl_network_structure"></a>
## Network Structure
For Reinforcement Learning, network input and output are required. In the case of Jetbot, the input is an image obtained by changing the size of the image taken by the camera to 84x84x3. And the output is an action consisting of a ball kick, stopping, moving forward, moving backward, turning left, turning right.

<img src="/assets/Jetbot_RL_Network.png">

<a name="rl_reward_setting"></a>
## Reward Setting

Moreover, it is necessary to define a reward for the training of Reinforcement Learning. At the very beginning, the camera is set to give a reward value of 1 when a soccer ball is detected by the camera.

```
r = 0
for obj in label:
    if obj == "sports ball":
        r = 1
        index = label.index(obj)
```

In the initial stage of Reinforcement Learning, training data is collected while performing random actions so that the agent can experience various situation. This part is for exploration of Reinforcement Learning.

[![Jetbot soccer test5](http://img.youtube.com/vi/og1vXFjZdbI/maxresdefault.jpg)](https://youtu.be/og1vXFjZdbI "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

After trarning, I can compare the behavior of Jetbot with the previous behavior to check Reinforcement Learning is effective.

<a name="rl_training_result"></a>
## Training result
When Reinforcement Learning works normally, after setting as above, Jetbot must see the direction where the soccer ball is located after a certain episode.I observe the behavior of Jetbot after Reinforcement Learning as a condition to receive a reward of 1 when soccer ball is detected.

[![First Jetbot Reinforcement Learning test1](https://img.youtube.com/vi/bbSEE9zK8a4/maxresdefault.jpg)](https://youtu.be/bbSEE9zK8a4 "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

<a name="rl_change_ball"></a>
## Changing ball size for well detection
I am able to confirm that there is a case where the total sum of reward do not increase even though the behavior of Jetbot is like finding a soccer ball.

<img src="/assets/soccer_ball_range.png" width="800">

Thus, I reconfirm the part that detects the soccer ball. The minimum distance for detection is confirmed around 1.3m. I think soccer ball size is needed bigger than now. 

<img src="/assets/middle_soccer_field.png" width="800">
Reflecting these circumstances, the size of the soccer field has been reduced to 2/3 due to the limited distance of the Jetbot camera. And the size of the soccer ball is made slightly larger so that it can be taken well by the camera.

<a name="rl_change_reward_setting"></a>
## Changing reward setting for accurate control
After adjusting the size of the field and the size of the ball to match Jetbot, the conditions for reward for Reinforcmenet Learning are slightly raised. When soccer ball enter the screen, Jetbot get a reward, but this time ball should be located around the center of screen.

```
r = 0
for obj in label:
  if obj == "sports ball":
    index = label.index(obj)
    ball_bbox_x1 = bbox[index][0]
    ball_bbox_x2 = bbox[index][2]
    ball_bbox_middle = (ball_bbox_x1 + ball_bbox_x2) / 2.0
    #print("ball_bbox_middle: " + str(ball_bbox_middle))
            
    if ( (ball_bbox_middle > 330) & (ball_bbox_middle < 390) ):
      r = 1
```

[![Second Jetbot Reinforcement Learning](https://img.youtube.com/vi/rW7n_T9aAQc/maxresdefault.jpg)](https://youtu.be/rW7n_T9aAQc "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

<a name="rl_second_training_result"></a>
## Training result after revision
After trarning about 12 hours, when I check a behavior of Jetbot, it seems to try to pick up a soccer ball inside the camera angle. Therefore, I can confirm that there is no problem with the Reinforcement Learning algorithm itself. 

However, Jetbot do not take a soccer ball exactly in the middle of the camera. Thus, I check a the conditions that give reward and find it is too wide and decide to narrow the scope.

[![Third Jetbot Reinforcement Learning](https://img.youtube.com/vi/vQseC4pwsBY/maxresdefault.jpg)](https://youtu.be/vQseC4pwsBY "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

I train after reducing the range displayed on the soccer ball camera that gives the reward. The results confirm that Jetbot shows the action to pick up the soccer ball in the middle of the camera. 

However, sometimes there is a phenomenon where Jetbot was standing in a place that was too long. Such a phenomenon is considered to be a phenomenon that occurs because there is no effect on the reward even if only the time has passed without doing anything.

<a name="real_jetbot_test"></a>
# Real Jetbot test
In addition to trarning Reinforcement Learning by simulation, I also test a the actual Jetbot to reduce the difference between simulation and actual.

<a name="control_original_jetbot"></a>
## Controlling original Jetbot
While trarning the simulation, the actual environment is constructed for using the trained model. I use various basic examples of Jetbot to check the operation of the robot in this environment. The carpet for soccer field looks like a little small. It is necessary to use the entire area including the white line.

[![First Jetbot real soccer test](https://img.youtube.com/vi/QxG27A8Z_L0/maxresdefault.jpg)](https://youtu.be/QxG27A8Z_L0
 "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

While creating the actual environment directly, I am able to discover difference between simulation and actual. The biggest one is that Jetbot should not collide with each other because the camera is exposed to the outside. It seems that camera will be broken at the time of collision.

<a name="deep_learning_model_jetbot"></a>
## Deep Learning model for Jetbot
In addition to creating a soccer field for Jetbot, I should also consider how to use the Tensorflow model trained in Gazebo. First of all, it is necessary to implement the input and output of the network of the neural network used in Gazebo as it is in the actual Jetbot. Thus, it will be possible to use the trained network without change.

<img src="/assets/env_change.png" width="800">

In particular, embedded boards such as Jetson Nano may not use common libraries can be used on general research PCs. Thus, this kind of library also needs to be changed. For example, I use cvlib when detecting a soccer ball in Gazebo, but this library correctly can not be installed in Jetson Nano.

```
import jetson.inference
import jetson.utils

b_channel, g_channel, r_channel = cv2.split(cv_image)
alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 # creating a dummy alpha channel image.
img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
in_arr = jetson.utils.cudaFromNumpy(img_BGRA)

overlay = "box,labels,conf"

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.3)
detections = net.Detect(in_arr, width, height, overlay)
```

First, jetson.inference and jetson.utils is needed to be imported in the python file. Next, change a format of camera image from numpy array to tensor using utils.cudaFromNumpy. Overlay parameter should be given to inference.detectNet function for boxing, labeling and getting confidence of detection.

<img src="/assets/jetbot_detect_result.jpg" width="800">

In fact, if you look at the results of a test using Jetbot, you can see that a fairly wide range of objects are detected and displayed with high accuracy.

<a name="transfer_tensorflow_model"></a>
## Transfering trained Tensorflow weight to Jetbot
And considering the limited Jetson Nano board GPU compared to a normal PC, it is necessary to use a trained model to optimize when inference. In the Jetson series, this can be done mainly using TensorRT. First the model is trained and saved using Gazebo and Reinforcmenet Learning on a general PC with GPU. Next, convert the weight of the saved model to TensorRT inference graph format, and use this to execute inference in session.

```
# Import TensorFlow and TensorRT
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np

# Inference with TF-TRT `MetaGraph` and checkpoint files workflow:
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        # First create a `Saver` object (for saving and rebuilding a
        # model) and import your `MetaGraphDef` protocol buffer into it:
        saver = tf.train.import_meta_graph("model-1.cptk.meta")
        # Then restore your training data from checkpoint files:
        
        checkpoint = tf.train.get_checkpoint_state("/home/jetbot/Notebooks/")
        input_checkpoint = checkpoint.model_checkpoint_path
        saver.restore(sess, input_checkpoint)
        
        #for op in graph.get_operations():
        #    print(op.name)
            
        # Finally, freeze the graph:
        your_outputs = ["main/ArgMax"]
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names=["main/ArgMax"])
        
        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=["main/ArgMax"],
            max_batch_size=1,
            max_workspace_size_bytes=50000,
            precision_mode="FP16")
        
        tf.import_graph_def(trt_graph, name='')
    
        x = graph.get_tensor_by_name('main/scalarInput:0')
        y = graph.get_tensor_by_name('main/ArgMax:0')
        trainLength = graph.get_tensor_by_name('main/trainLength:0')
        batch_size = graph.get_tensor_by_name('main/batch_size:0')
    
        input_test = np.empty([84, 84, 3], dtype=float)
        input_test_flatten = input_test.flatten()
        
        y_out = sess.run(y, feed_dict={
            x: [input_test_flatten],
            trainLength: 1,
            batch_size: 1
        })
        
        print(y_out)
```

