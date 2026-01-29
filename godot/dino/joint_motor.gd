extends HingeJoint3D

@export var max_tor = 1;

var pow = 1



var first_obg : RigidBody3D
var second_obg : RigidBody3D

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	first_obg = get_node(node_a) as RigidBody3D
	
	second_obg = get_node(node_b) as RigidBody3D


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pow = clamp(pow,-1,1)
	
	
	var axis = second_obg.global_transform.basis.z
	second_obg.apply_torque(axis  * pow * max_tor)
	first_obg.apply_torque(axis * -1 * pow * max_tor)
	
func get_angle():
	var rot1 = first_obg.rotation.z
	
	var rot2 = second_obg.rotation.z

	return  rot1 - rot2
func get_vel():
	
	
	
	
	var rot1 = (first_obg.angular_velocity* second_obg.global_basis).z
	
	var rot2 = (second_obg.angular_velocity* second_obg.global_basis).z

	return  (rot1 - rot2)
	
	
func  _input(event: InputEvent) -> void:
	if event is InputEventKey and event.is_pressed():
		#pow = -pow
		pass
