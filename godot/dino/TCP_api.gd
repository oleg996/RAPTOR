extends Node

var stream = StreamPeerTCP.new()

var data_to_send = [1.2,1.3,1.5,1.5]


var data_to_recive = []


var debug = false

var in_lenght = 36
#4 bytes per digitt + 1 for reset?
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass
	if not debug:
		conncet()
		while stream.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			stream.poll()
		print('Connected')

func conncet():
	stream.connect_to_host("127.0.0.1",10003)


	
func send():
	var araay =  PackedFloat32Array(data_to_send)
	
	stream.put_data(araay.to_byte_array())
	
	
func recive():
	var values = stream.get_data(in_lenght)[1]
	
	
	values = PackedByteArray(values).to_float32_array()
	
	data_to_recive = values
	
