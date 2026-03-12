extends Joint3D

@export var max_tor = 1;


var pow = 1

var max_vel = 7

var first_obg : RigidBody3D
var second_obg : RigidBody3D

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	first_obg = get_node(node_a) as RigidBody3D
	
	second_obg = get_node(node_b) as RigidBody3D
	
	max_tor = max_tor



func _physics_process(delta: float) -> void:
	pow = clamp(pow,-1,1)

	#var target_pow = pow*max_tor+get_vel()*max_tor/16
	
	var power_tor = pow * max_tor
	
	var emf = (get_vel()/max_vel)*max_tor
	var target_pow = power_tor + emf
	
	var axis = second_obg.global_transform.basis.z
	second_obg.apply_torque(axis  * target_pow)
	first_obg.apply_torque(axis * -1 * target_pow)
	
func get_angle():
	var rot1 = first_obg.rotation.z
	
	var rot2 = second_obg.rotation.z

	return  rot1 - rot2
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
		pow = -pow
		pass
