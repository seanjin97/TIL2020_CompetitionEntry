import socket
from threading import Thread
import cv2
import time


def findrobotIP(): # listen on UDP broadcast for robot's IP. 
	ipbroadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	ipbroadcast_socket.bind(('', 40926))
	broadcast_data, broadcast_addr = ipbroadcast_socket.recvfrom(1024)
	robot_addr = broadcast_addr[0]
	print(f'\tdata sent is {broadcast_data} from {broadcast_addr}')
	return robot_addr # a string


class Robot():
	def __init__(self, robot_ip):
		self.robot_ip = robot_ip
		self.commandsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.commandsocket.connect((self.robot_ip, 40923))
		self.commandsocket.sendall('command;'.encode())
		commandreply = self.commandsocket.recv(1024)
		print(f'Command socket: {commandreply.decode("utf-8")}')

		self.cap = None
		self.frame = None
	
	def _sendcommand(self, cmdstring, echo=True): 
		assert isinstance(cmdstring, str)
		cmdstring = cmdstring+';'
		self.commandsocket.sendall(cmdstring.encode())
		commandreply = self.commandsocket.recv(1024)
		commandreply = commandreply.decode("utf-8")
		if echo:
			print(f'\treply for {cmdstring}: {commandreply}')
		return commandreply

	def startvideo(self):
		self._sendcommand('stream on')
		self.videothread = Thread(target=self._receive_video_thread, daemon=True)
		self.videothread.start() # updates self.frame
		if self.videothread.isAlive():
			print('videothread started!')

	def _receive_video_thread(self):
		"""
		Listens for video streaming (raw h264).
		Runs as a thread, sets self.frame to the most recent frame captured. frame processing and imshow should be in main loop
		"""
		self.cap = cv2.VideoCapture(f'tcp://@{self.robot_ip}:40921')
		while self.cap.isOpened():
			ret, self.frame = self.cap.read()


# ROBOT COMMANDS
	def move(self, inputstring):
		'''
		input: x dist y dist z angle vxy m/s Vz degree/s
		moves chassis by distance
		e.g. move('x 0.5') to move forward by 0.5m
		'''
		command = 'chassis move ' + inputstring
		return self._sendcommand(command)

	def rotate(self, inputstring):
		'''
		input: string in degrees
		rotate CW
		'''
		command = 'chassis move z ' + inputstring
		return self._sendcommand(command)

	def right(self, inputstring):
		'''
		input: integer in m/s
		constantly strafe right
		'''
		command = 'chassis speed y ' + inputstring
		return self._sendcommand(command)

	def forward(self, inputstring):
		'''
		input: integer in m per s
		constantly move forward
		'''
		command = 'chassis speed x ' + inputstring
		return self._sendcommand(command)

	def movearm(self, inputstring):
		"""
		input: x {val} y {val}
		"""
		command = 'robotic_arm move ' + inputstring
		return self._sendcommand(command)
		
	def openarm(self):
		return self._sendcommand('robotic_gripper open 1')

	def closearm(self):
		return self._sendcommand('robotic_gripper close 1')

	def rotategimbal(self, inputstring):
		'''
		relative to current position
		p angle y angle vp speed vy speed

		'''
		command = 'gimbal move ' + inputstring
		return self._sendcommand(command)

	def rotategimbalto(self):
		'''
		relative to initial position
		p angle y angle vp speed vy speed
		'''
		command = 'gimbal moveto ' + inputstring
		return self._sendcommand(command)

	def stop(self):
		self._sendcommand('chassis move x 0')
		self._sendcommand('gimbal speed p 0 y 0') # stops if in speed. gimbal move will finish
		self._sendcommand('robotic_arm stop') 

	def center(self):
		self._sendcommand('gimbal moveto p 0 y 0')
		self._sendcommand('robotic_arm recenter') 

	def exit(self): # call this to stop robot and bring back to center position before exiting
		print('Exiting...')
		self.stop()
		self.center()
		time.sleep(2)
		if self.cap is not None:
			self.cap.release()
		self._sendcommand('stream off')
		self._sendcommand('quit')
		self.commandsocket.close()
		print('All done')
