"""
Scipt to generate the dataset for `Proprioceptive Character Recognition` task.

"""
import numpy as np
import opensim as osim
import scipy.optimize


class Arm:

    def __init__(self, q=None, q0=None, L=None):
        # Initial joint angles
        self.q = np.array([10, 10, 10, 10]) if q is None else q
        # Default arm position, set to last position in the trajectory.
        self.q0 = np.array([50, 40, 20, 75]) if q0 is None else q0
        # Lengths of humerus and radius (in cm).
        self.L = np.array([33, 26]) if L is None else L

        # Maximum and minimum angles. [in degrees]
        self.min_angles = np.array([-90, 0, -90, 0])
        self.max_angles = np.array([130, 180, 20, 130])

    def get_xyz(self, q=None):
        """Implements forward kinematics:
        Returns the end-effector coordinates (euclidean) for a given set of joint
        angle values.
        Inputs :
        Returns :
        """
        if q is None:
            q = self.q

        # Define rotation matrices about the shoulder and elbow.
        # Translations for the shoulder frame will be introduced later.
        def shoulder_rotation(elv_angle, shoulder_elv, shoulder_rot):
            return roty(elv_angle).dot(rotz(shoulder_elv)).dot(roty(-elv_angle)).dot(roty(shoulder_rot))

        def elbow_rotation(elbow_flexion):
            return rotx(elbow_flexion)

        # Unpack variables
        elv_angle, shoulder_elv, shoulder_rot, elbow_flexion = q
        upperarm_length, forearm_length = self.L

        # Define initial joint locations:
        origin = np.array([0, 0, 0])
        elbow = np.array([0, -upperarm_length, 0])
        hand = np.array([0, -forearm_length, 0])

        new_elbow_loc = shoulder_rotation(
            elv_angle, shoulder_elv, shoulder_rot).dot(elbow)
        new_hand_loc = shoulder_rotation(elv_angle, shoulder_elv, shoulder_rot)\
            .dot(elbow_rotation(elbow_flexion).dot(hand) + elbow)

        link_pos = np.column_stack((origin, new_elbow_loc, new_hand_loc))

        return new_hand_loc, link_pos

    def inv_kin(self, xyz):
        """Implements inverse kinematics:
        Given an xyz position of the hand, return a set of joint angles (q)
        using constraint based minimization. Constraint is to match hand xyz and
        minimize the distance of each joint from it's default position (q0).
        Inputs :
        Returns :
        """

        def distance_to_default(q, *args):
            return np.linalg.norm(q - np.asarray(self.q0))

        def pos_constraint(q, xyz):
            return np.linalg.norm(self.get_xyz(q=q)[0] - xyz)

        def joint_limits_upper_constraint(q, xyz):
            return self.max_angles - q

        def joint_limits_lower_constraint(q, xyz):
            return q - self.min_angles

        return scipy.optimize.fmin_slsqp(
            func=distance_to_default, x0=self.q, eqcons=[pos_constraint],
            ieqcons=[joint_limits_upper_constraint, joint_limits_lower_constraint],
            args=(xyz,), iprint=0)


# Auxiliary function definitions:
# Define rotation matrices which take angle inputs in degrees.
def rotx(angle):
    angle = angle*np.pi/180
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])


def roty(angle):
    angle = angle*np.pi/180
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])


def rotz(angle):
    angle = angle*np.pi/180
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


# Inverse kinematics and muscle length generation utilities

def make_joint_config(trajectory):
    """Compute joint configurations of a 4Dof Arm given end-effector trajectory in {W}.

    Returns
    -------
    joint_coordinates: np.array, 4 joint angles at each of T time points
    error: np.float, error in tracing out the given end-effector trajectory

    """
    myarm = Arm()
    shoulder_to_world = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    # Inverse Kinematics to obtain joint configurations for the character trajectory
    # Project character trajectories into {S}
    char_traj_s = (shoulder_to_world.T).dot(trajectory)
    traj_length = char_traj_s.shape[1]
    joint_trajectory = np.zeros((4, traj_length))

    # For each point in the trajectory derive the joint angle configuration
    # After finding the joint configuration for a particular point, change q0
    # to the current joint configuration!
    error = 0.
    for i in range(traj_length):
        dest = char_traj_s[:, i]
        joint_config = myarm.inv_kin(xyz=dest)
        myarm.q0 = joint_config
        joint_trajectory[:, i] = joint_config
        error += np.linalg.norm(dest - myarm.get_xyz(joint_config)[0])

    return joint_trajectory, error


def compute_jerk(joint_trajectory):
    """Compute the jerk in joint space for the obtained joint configurations.

    Returns
    -------
    jerk : np.array, [T,] array of jerk for a given trajectory

    """
    joint_vel = np.gradient(joint_trajectory, axis=1)
    joint_acc = np.gradient(joint_vel, axis=1)
    joint_jerk = np.gradient(joint_acc, axis=1)
    jerk = np.linalg.norm(joint_jerk)
    return jerk


def make_muscle_config(model, joint_trajectory):
    """Compute muscle configurations of a given opensim model to given coordinate (joint angle) trajectories.

    Arguments
    ---------
    model : opensim model object, the MoBL Dynamic Arm.
    joint_trajectory : np.array, shape=(4, T) joint angles at each of T time points

    Returns
    -------
    muscle_length_configurations : np.array, shape=(25, T) muscle lengths at each of T time points

    """
    init_state = model.initSystem()
    model.equilibrateMuscles(init_state)

    # Prepare for simulation
    num_coords, num_timepoints = joint_trajectory.shape
    muscle_set = model.getMuscles() # returns a Set<Muscles> object
    num_muscles = muscle_set.getSize()

    # Set the order of the coordinates
    coord_order = ['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion']

    # For each time step of the trajectory, compute equibrium muscle states
    # Create a dictionary of muscle configurations
    muscle_config = {}
    for i in range(num_muscles):
        muscle_config[muscle_set.get(i).getName()] = np.zeros(num_timepoints)

    for timepoint in range(num_timepoints):
        for i in range(num_coords):
            model.getCoordinateSet().get(coord_order[i]).\
            setValue(init_state, np.pi*(joint_trajectory[i, timepoint] / 180))
        model.equilibrateMuscles(init_state)
        for muscle_num in range(num_muscles):
            muscle = muscle_set.get(muscle_num)
            name = muscle.getName()
            muscle_config[name][timepoint] = muscle.getFiberLength(init_state)*1000 # change to mm

    # Delete muscles that are not of interest to us
    shoulder_muscles = ['CORB', 'DELT1', 'DELT2', 'DELT3', 'INFSP', 'LAT1', 'LAT2', 'LAT3',
                        'PECM1', 'PECM2', 'PECM3', 'SUBSC', 'SUPSP', 'TMAJ', 'TMIN']
    elbow_muscles = ['ANC', 'BIClong', 'BICshort', 'BRA', 'BRD', 'ECRL', 'PT', 'TRIlat',
                     'TRIlong', 'TRImed']

    for i in list(muscle_config.keys()):
        if not (i in shoulder_muscles or i in elbow_muscles):
            del muscle_config[i]

    muscle_length_configurations = make_muscle_matrix(muscle_config)

    return muscle_length_configurations


def make_muscle_matrix(muscle_config):
    # Arrange muscles configurations in a 25xT matrix, given a dictionary of muscle configurations.
    order = ['CORB', 'DELT1', 'DELT2', 'DELT3', 'INFSP', 'LAT1', 'LAT2', 'LAT3', 'PECM1',
             'PECM2', 'PECM3', 'SUBSC', 'SUPSP', 'TMAJ', 'TMIN', 'ANC', 'BIClong', 'BICshort',
             'BRA', 'BRD', 'ECRL', 'PT', 'TRIlat', 'TRIlong', 'TRImed']
    mconf = [muscle_config[i] for i in order]
    return np.asarray(mconf)

def signpow(a,b): return np.sign(a)*(np.abs(a)**b)

def make_spindle_coords(muscle_traj):
    stretch = np.gradient(muscle_traj, 1, axis=1)
    stretch_vel = np.gradient(muscle_traj, 0.015, axis=1)
    p_rate = 2*stretch + 4.3*signpow(stretch_vel, 0.6)
    return p_rate

def start_end_choice(traj):
    true_traj = traj[:, np.all(~np.isnan(traj), axis=0)]
    room = 320 - true_traj.shape[1]
    start_idx = np.random.randint(room)
    end_idx = start_idx + true_traj.shape[1]
    return start_idx, end_idx

def apply_shuffle(traj, start_idx, end_idx):
    true_traj = traj[:, np.all(~np.isnan(traj), axis=0)]
    mytraj = np.zeros((true_traj.shape[0], 320))
    mytraj[:, start_idx:end_idx] = true_traj
    mytraj[:, :start_idx] = true_traj[:, 0][:, None]
    mytraj[:, end_idx:] = true_traj[:, -1][:, None]
    return mytraj

def add_noise(mconf, factor):
    noisy_mconf = mconf + factor*mconf.std(axis=1)[:, None]*np.random.randn(*mconf.shape)
    return noisy_mconf