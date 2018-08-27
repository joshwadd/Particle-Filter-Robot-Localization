/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // number of particles generated requires tuning
    num_particles = 100;

    default_random_engine gen;

    weights.resize(num_particles);
    particles.resize(num_particles);

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    int id = 0;
    for(auto& particle : particles){
        particle.id = id++;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;

    // noise generating distributions
    // Add gussian distributed noise to the process model to account for
    // the uncertainty in the control input
    // (velocity and steering angle (aka yaw rate), being the control inputs)
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    /*
     * Use bicycle process model to predict the state of each particle
     * having recieved the noisy velocity, steering inputs.
     */

    for(auto& particle : particles){

        if (abs(yaw_rate) != 0){
            particle.x += (velocity/yaw_rate) * (sin(particle.theta + (yaw_rate * delta_t)) - sin(particle.theta));
            particle.y += (velocity/yaw_rate) * (cos(particle.theta) - cos(particle.theta + (yaw_rate * delta_t)));
            particle.theta += yaw_rate * delta_t;

        } else{
            particle.x += velocity * delta_t * cos(particle.theta);
            particle.y += velocity * delta_t * sin(particle.theta);
            // theta remains the same due to constant grad
        }

        particle.x += dist_x(gen);
        particle.y += dist_y(gen);
        particle.theta += dist_theta(gen);

    }




}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


    for(auto& obs : observations){ // loop through all observations
        double current_minimum_distance = std::numeric_limits<double>::max();

        for( auto& pred : predicted ){ // loop through all predictions

            double euclidean_distance = dist(obs.x, obs.y, pred.x, pred.y);
            if(euclidean_distance < current_minimum_distance){
                current_minimum_distance = euclidean_distance;
                obs.id = pred.id;
            }
        }
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


    // pre compute some constant values
    const double kNormalizingConstant = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    const double kDenominator_x = 2 * std_landmark[0] * std_landmark[0];
    const double kDenominator_y = 2 * std_landmark[1] * std_landmark[1];


    for( int i = 0; i < num_particles; i++) { // iterate through all particles

        // Find the landmarks within the sensor range of each particle
        vector<LandmarkObs> proximal_landmarks;
        for(auto& landmark : map_landmarks.landmark_list){
            double distance = dist(landmark.x_f, landmark.y_f, particles[i].x, particles[i].y);
            if(distance < sensor_range){
               LandmarkObs new_landmark;

               new_landmark.id = landmark.id_i;
               new_landmark.x = landmark.x_f;
               new_landmark.y = landmark.y_f;
               proximal_landmarks.push_back(new_landmark);
            }
        }


        // Transform each observation from the vehcile into the map cordinate frame
        vector<LandmarkObs> observations_map_frame;
        for(auto& observation : observations){
            LandmarkObs transformed_observation;
            transformed_observation.id = observation.id;
            transformed_observation.x = observation.x * cos(particles[i].theta) - observation.y * sin(particles[i].theta) + particles[i].x;
            transformed_observation.y = observation.x * sin(particles[i].theta) + observation.y * cos(particles[i].theta) + particles[i].y;
            observations_map_frame.push_back(transformed_observation);
        }


        // Do Nearest Neighbour data assoiation
        dataAssociation(proximal_landmarks, observations_map_frame);

        // reset the weight value
        particles[i].weight = 1.0;

        // variables for particle assoiation debugging
        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;

        // compute and update the new weight values, using multivariate guassian distribution
        for(auto& obs_map_frame : observations_map_frame){

            int observations_landmark_id = obs_map_frame.id;
            Map::single_landmark_s landmark = map_landmarks.landmark_list[observations_landmark_id - 1];

            double delta_x = obs_map_frame.x - landmark.x_f;
            double delta_y = obs_map_frame.y - landmark.y_f;

            double weight = kNormalizingConstant * exp(-((delta_x*delta_x)/kDenominator_x +  (delta_y*delta_y)/kDenominator_y) );

            if(weight == 0){
                particles[i].weight *= 0.00001;
            } else{
                particles[i].weight *= weight;
            }

            associations.push_back(obs_map_frame.id);
            sense_x.push_back(obs_map_frame.x);
            sense_y.push_back(obs_map_frame.y);

        }

        weights.push_back(particles[i].weight);
        SetAssociations(particles[i], associations, sense_x, sense_y);

    }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<Particle> resampled_particles;

    //create discrete distibution from particles weights
    random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> distribution(weights.begin(), weights.end());

    for( int i = 0; i < num_particles; i++){
        const int index = distribution(gen);
        const Particle new_particle = particles[index];
        resampled_particles.push_back(new_particle);
    }

    particles.clear();
    weights.clear();

    particles = resampled_particles;


}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();


    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
