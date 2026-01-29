extends RigidBody3D
var pre_tr = null

var to_reset = false

var random_vel = 2

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pre_tr = global_transform
	

	
	
func reset():
	to_reset = true

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
func _integrate_forces(state: PhysicsDirectBodyState3D) -> void:	
	if to_reset:
		state.transform = pre_tr
		state.linear_velocity = random_vel * Vector3(randf_range(-1,1),randf_range(-1,1),randf_range(-1,1))
		state.angular_velocity = random_vel * Vector3(randf_range(-1,1),randf_range(-1,1),randf_range(-1,1))
		to_reset = false
