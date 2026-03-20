extends Joint3D

@export var max_tor = 1;


var pow = 0.9 

var max_vel = 10

var first_obg : RigidBody3D
var second_obg : RigidBody3D

var zero = 0

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	first_obg = get_node(node_a) as RigidBody3D
	
	second_obg = get_node(node_b) as RigidBody3D
	
	max_tor = max_tor*1.5

	zero = get_angle()

func _physics_process(delta: float) -> void:
	pow = clamp(pow,-1,1)

	var err =(pow- get_norm_ang())*10
	err = clamp(err,-1,1)
	var power_tor = err* max_tor
	
	var emf = (get_vel()/max_vel)*max_tor
	var target_pow = power_tor + emf
	
	var axis = second_obg.global_transform.basis.z
	second_obg.apply_torque(axis  * target_pow)
	first_obg.apply_torque(axis * -1 * target_pow)
	
func get_angle():
	
	var vec1 = first_obg.global_basis.x
	
	var vec2 = second_obg.global_basis.x
	
	var dir = second_obg.global_basis.z
	
	var ang = vec1.signed_angle_to(vec2,dir)-zero
	

	return ang 
	
func get_norm_ang():
	return get_angle() /get("angular_limit/upper")
	
func get_vel():
	
	
	
	
	var rot1 = (first_obg.angular_velocity* second_obg.global_basis).z
	
	var rot2 = (second_obg.angular_velocity* second_obg.global_basis).z

	return  (rot1 - rot2)

var prev_vel = 0

func get_accel():
	var cur_vel = get_vel()
	var acc = cur_vel -prev_vel	
	prev_vel = cur_vel
	return acc
func  _input(event: InputEvent) -> void:
	if event is InputEventKey and event.is_pressed():
		#pow = -pow
		pass
