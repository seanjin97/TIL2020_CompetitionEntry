import cv2
from threading import Thread
import socket
import time

class Tello():
	def __init__(self, minheight=20, maxheight=200):
		self.commandsocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.commandsocket.bind(('', 8889))
		self.commandsocket.settimeout(8) # wait for reply in recvfrom
		self._sendcommand('command')

		self.statesocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		print('Creating state socket')

		self.frame = None
		self.airborne = False
		self.height = 0 
		self.minheight = minheight
		self.maxheight = maxheight
	
	def _sendcommand(self, cmdstring, tries=3):
		assert isinstance(cmdstring, str)
		for i in range(tries):
			self.commandsocket.sendto(cmdstring.encode(), ('192.168.10.1', 8889))
			try:
				commandreply, addr = self.commandsocket.recvfrom(1024)
				reply = commandreply.decode("utf-8")
				print(f'\treply for {cmdstring}: {reply}')
				break
			except socket.timeout:
				print(f'{cmdstring}, Attempt {i+1} out of {tries} failed')
				reply = f'{cmdstring} failed'
		return reply

	def startvideo(self):
		self._sendcommand('streamon')
		self.videothread = Thread(target=self._receive_video_thread, daemon=True)
		self.videothread.start()
		if self.videothread.isAlive():
			print('Video thread started!')

	def _receive_video_thread(self):
		self.cap = cv2.VideoCapture('udp://0.0.0.0:11111')
		while self.cap.isOpened():
			ret, self.frame = self.cap.read()

	def startstates(self):
		self.statesocket.bind(('',8890))
		self.statethread = Thread(target=self._receive_state_thread, daemon=True)
		self.statethread.start()
		if self.statethread.isAlive():
			print('State thread started!')
	
	def _receive_state_thread(self):
		"""
		Listens for states. currently only taking height
		"""
		while True:
			states, addr = self.statesocket.recvfrom(1024)
			states = states.decode("utf-8")
			position = states.find(';h:')
			height = states[position+3:].split(';')[0]
			self.height = int(height)

	def start_pad_det(self):
		self._sendcommand('mon')
		self._sendcommand('mdirection 0')

	def act(self, k): # press these keys at opened cv2 window  to manually adjust drone
		if k == ord('t'):
			takeoffreply = self._sendcommand('takeoff')
			if takeoffreply == 'ok':
				self.airborne = True
		if k == ord('l'):
			landreply = self._sendcommand('land') # still sent if not airborne. should see error.
			if landreply == 'ok': 
				self.airborne = False

		if k == ord('w'):
			self._sendcommand('forward 20')
		if k == ord('a'):
			self._sendcommand('left 20')
		if k == ord('s'):
			self._sendcommand('back 20')
		if k == ord('d'):
			self._sendcommand('right 20')
		if k == ord('q'):
			self._sendcommand('ccw 10')
		if k == ord('e'):
			self._sendcommand('cw 10')
		if k == ord('u'): # ascends
			if (self.maxheight - self.height) >= 20:
				self._sendcommand(f'up 20')
			else:
				print(f'Max height is {self.maxheight}')
			print(f'Current height is {self.height}')
		if k == ord('j'): # descends
			if (self.height - self.minheight) >= 20:
				self._sendcommand(f'down 20')
			else:
				print(f'Min height is {self.minheight}')
			print(f'Current height is {self.height}')

	def exit(self):
		self._sendcommand('land') # should be sent regardless of airborne to ensure it lands before prog ends
		time.sleep(2)
		self._sendcommand('streamoff')
		self._sendcommand('moff')
		self.commandsocket.close()

