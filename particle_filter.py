from grid import CozGrid
from particle import Particle
from utils import grid_distance, rotate_point, diff_heading_deg, add_odometry_noise
import setting
import math
import numpy as np
import pdb


def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*
        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used for boundary checking
        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = []
    if odom is None:
        return particles
    dX, dY, dH = odom[0], odom[1], odom[2] #raw readings

    for particle in particles:
        x, y, h = particle.xyh
        dX, dY = rotate_point(dX, dY, h) #relative to inital angle
        rotatedX, rotatedY, rotatedH = (x + dX, y + dY, h + dH) #add changes in local reference frame
        finalX, finalY, finalH = add_odometry_noise((rotatedX, rotatedY, rotatedH),
                                                    setting.ODOM_HEAD_SIGMA,
                                                    setting.ODOM_TRANS_SIGMA) #add odometry noise
        finalH = finalH % 360
        motion_particles.append(Particle(finalX, finalY, finalH))

    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before measurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    measured_particles = []
    weights = []
    weightTotal = 0
    randomSamples = 30
    removeIndex = []
    i = 0
    min_weight_threshold = 0

    if measured_marker_list is None:
        return particles

    #assign weights
    for particle in particles:
        x, y = particle.xy
        if not grid.is_free(x, y) or (measured_marker_list and not particle.read_markers(grid)):
            weights.append(0)
        else:
            weight = getWeight(particle, measured_marker_list, grid)
            if weight < min_weight_threshold:
                randomSamples += 0
                removeIndex.append(i)
            else:
                weightTotal += weight
            weights.append(weight)
        i += 1
    for index in reversed(removeIndex):
        del particles[index]
        del weights[index]

    #normalize
    probabilities = []
    if weightTotal == 0:
        for weight in weights:
            probabilities.append(1 / len(weights))
    else:
        for weight in weights:
            probabilities.append(weight / weightTotal)

    #resample
    resampled_particles = np.random.choice(particles, len(weights) - randomSamples,
                                           True, probabilities)
    random_particles = Particle.create_random(randomSamples, grid)
    measured_particles = resampled_particles.tolist() + random_particles

    return measured_particles

def getWeight(particle, robot_marker_list, grid):
    particle_marker_list = particle.read_markers(grid) #only includes markers in robot FOV
    pairs = []
    for robot_marker in robot_marker_list: #match markers to best fit pairs
        xr, yr, _ = robot_marker
        bestDist = float('inf')
        if len(particle_marker_list) == 0:
            break
        bestParticleMarker = particle_marker_list[0]
        for i in range(0, len(particle_marker_list)):
            xp, yp, _ = particle_marker_list[i]
            currDist = grid_distance(xr, yr, xp, yp)
            if currDist < bestDist:
                bestParticleMarker = particle_marker_list[i]
                bestDist = currDist
        pairs.append((robot_marker, bestParticleMarker))
        particle_marker_list.remove(bestParticleMarker)
    p = 1
    for robot_marker, particle_marker in pairs:
        p *= normalDistribution(robot_marker, particle_marker)
    return p

def normalDistribution(marker1, marker2):
    x1, y1, h1 = marker1
    x2, y2, h2 = marker2
    distDiff = grid_distance(x1, y1, x2, y2)
    angleDiff = diff_heading_deg(h1, h2)
    distConstant = -0.5 / (setting.MARKER_TRANS_SIGMA ** 2)
    angleConstant = -0.5 / (setting.MARKER_ROT_SIGMA ** 2)
    return math.exp(distDiff ** 2 * distConstant +
                    angleDiff ** 2 * angleConstant)