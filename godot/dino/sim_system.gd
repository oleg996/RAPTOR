extends Node3D


var reset_frame = false

var parts 


var counter = 0


var standard_FPS = 60

var speed_Factor = 1

var pad_x = 0
var pad_y = 0

var sim_time =0

var sigma = 1

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	Engine.time_scale = speed_Factor

	Engine.max_fps = standard_FPS*speed_Factor
	Engine.physics_ticks_per_second = standard_FPS*speed_Factor*3
	
	
	parts = [$dino/body,$dino/bleg1,$dino/bleg2,$dino/bleg3,$dino/fleg1,$dino/fleg2,$dino/fleg3,$dino/tail1,$dino/neck1]

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	
	
	if not TcpApi.debug:
		if reset_frame:
			await get_tree().process_frame
			#set_rand_dir()
			reset_frame = false
		else:
			tick()
			sim_time += 1
			if sim_time > 500:
				sim_time = 0
				#set_rand_dir()
func tick():
	TcpApi.recive()
	
	
	
	if TcpApi.data_to_recive[0] == -1:	
		reset()
	else :
		var actions = []
		
		for i in range(8):
			TcpApi.data_to_recive.remove_at(0)
			actions.append(TcpApi.data_to_recive[0])
		
		perform_actions(actions)
	
	
	var obs = perform_observarions()
	var rev = calculate_revard()
	var data = [rev,0]
	if is_terminal():
		data[1] = -1
		#data[0] = -200
	
	data.append_array(obs) 
	
	TcpApi.data_to_send = data
	
	TcpApi.send()
	
func perform_actions(actions):
	
	$dino/body_bleg1.pow = actions[0]
	$dino/bleg1/bleg1_bleg2.pow = actions[1]
	$dino/bleg2/bleg2_bleg3.pow = actions[2]
	
	
	$dino/body_fleg1.pow = actions[3]
	$dino/fleg1/fleg1_fleg2.pow = actions[4]
	$dino/fleg2/fleg2_fleg3.pow = actions[5]
	
	$dino/body_tail1.pow = actions[6]
	
	$dino/body_neck1.pow = actions[7]
	
func perform_observarions():
	var obss = []
	
	
	var rot = $dino/body.global_transform.basis.rotated(Vector3.UP,-$dino/body.global_rotation.y).y
	
	obss.append(rot.x)
	
	obss.append(rot.y)
	
	obss.append(rot.z)
	
	var lvel = $dino/body.linear_velocity * $dino/body.global_basis
	
	obss.append(lvel.x)
	
	obss.append(lvel.y)
	
	obss.append(lvel.z)
	
	var avel = $dino/body.angular_velocity * $dino/body.global_basis
	
	obss.append(avel.x)
	
	obss.append(avel.y)
	
	obss.append(avel.z)
	
	
	add_angle(obss,$dino/body_bleg1.get_angle())
	add_angle(obss,$dino/bleg1/bleg1_bleg2.get_angle())
	add_angle(obss,$dino/bleg2/bleg2_bleg3.get_angle())
	
	add_angle(obss,$dino/body_fleg1.get_angle())
	add_angle(obss,$dino/fleg1/fleg1_fleg2.get_angle())
	add_angle(obss,$dino/fleg2/fleg2_fleg3.get_angle())
	
	add_angle(obss,$dino/body_tail1.get_angle())
	
	add_angle(obss,$dino/body_neck1.get_angle())
	
	
	
	obss.append($dino/body_bleg1.get_vel())
	obss.append($dino/bleg1/bleg1_bleg2.get_vel())
	obss.append($dino/bleg2/bleg2_bleg3.get_vel())
	
	obss.append($dino/body_fleg1.get_vel())
	obss.append($dino/fleg1/fleg1_fleg2.get_vel())
	obss.append($dino/fleg2/fleg2_fleg3.get_vel())
	
	obss.append($dino/body_tail1.get_vel())
	
	obss.append($dino/body_neck1.get_vel())
	
	
	
	
	
	obss.append(1 if $dino/bleg3.get_contact_count() > 0 else 0)
	
	obss.append(1 if $dino/fleg3.get_contact_count() > 0 else 0)
	
	#obss.append($dino/body.position.y)
	
	#obss.append($dino/body.global_transform.basis.x.z)
	
	var inp = keyboard()
	obss.append(inp[0])
	obss.append(inp[1])
	print(inp)
	return obss
	
func calculate_revard():
	
	var en_pen = pow($dino/body_bleg1.pow ,2) + pow($dino/bleg1/bleg1_bleg2.pow ,2) + pow($dino/bleg2/bleg2_bleg3.pow ,2)
	en_pen += 	pow($dino/body_fleg1.pow,2) + pow($dino/fleg1/fleg1_fleg2.pow,2) + pow($dino/fleg2/fleg2_fleg3.pow,2)
	en_pen += 	pow($dino/body_tail1.pow,2) + pow($dino/body_neck1.pow,2)
	
	var up_rew = $dino/body.global_transform.basis.y.y
	
	#var mov = $dino/body.linear_velocity.x - abs($dino/body.linear_velocity.z)
	
	var mov = calc_reward_for_walk()
	
	var col = 1 if is_colides() else 0
	
	var rev = mov+up_rew - en_pen * 0.0 - col                                           
	return rev
	
func reset():
	for p in parts:
		p.reset()
	reset_frame = true
	
	


func is_colides():
	var end = $dino/body.get_contact_count() > 0 or  $dino/bleg1.get_contact_count() > 0 or  $dino/bleg2.get_contact_count() > 0 or  $dino/fleg1.get_contact_count() > 0 or  $dino/fleg2.get_contact_count() > 0
	
	end = end or $dino/tail1.get_contact_count() > 0  or $dino/neck1.get_contact_count() > 0 
	
	return end
func is_terminal():
	var z_end = $dino/body.position.y < 0.20
	
	
	return z_end
	
func add_angle(arr, angle):
	arr.append(cos(angle))
	arr.append(sin(angle))
	
var ldelta = 1

func  _physics_process(delta: float) -> void:
	ldelta = delta

func FPS_TIMER() -> void:
	print("FPS:",Engine.get_frames_per_second())
	print("FTPS:",1/ldelta * speed_Factor)
	
func  _input(event: InputEvent) -> void:
	if event is InputEventKey and event.is_pressed():
		#reset()
		#print(perform_observarions())	
		pass
		
func calc_values():
	if pad_x == 0 and pad_y == 0:
		return [0,0]
	else :
		var angl = atan2(pad_x,pad_y)
		var vel = sqrt(pow(pad_x,2)+pow(pad_y,2))
		
		return [vel,angl]
		
func process_for_nn():
	var angl = calc_values()
	
	var angl_err = angl[1]-$dino/body.global_rotation.y
	
	return [angl_err,angl[0]]
	
	
func keyboard():
	var out = [0.0,0.0]
	out[1] = clamp(Input.get_axis("back","forw"),0,1)
	
	out[0] = clamp(Input.get_axis("left","right"),-1,1)*0.5
	
	return out
func calc_reward_for_walk():
	
	var targ = calc_values()
	
	var lvel = $dino/body.linear_velocity * $dino/body.global_basis
	
	var angle = targ[1]-$dino/body.global_rotation.y
	
	var vele = lvel.x-targ[0]*3
	
	var rew_hea = exp(-(pow(vele,2))/4)
	
	var rew_angl = exp(-(pow(angle,2))/2)

	return rew_angl + rew_hea
	
func set_rand_dir():
	if randf() > 0.5:
		pad_x =0
		pad_y =0
	else:
		pad_x =0#randf_range(-1,1)
		pad_y =0    #randf_range(-1,1)
	print("set pad to",pad_x,"|",pad_y)
