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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 10;

	default_random_engine gen;
	normal_distribution<double> x_init_dist(x, std[0]);
	normal_distribution<double> y_init_dist(y, std[1]);
	normal_distribution<double> theta_init_dist(theta, std[2]);
	
	for (int i = 0; i < num_particles; ++i)
	{
	    double p_x = x_init_dist(gen);
		double p_y = y_init_dist(gen);
		double p_theta = theta_init_dist(gen);
		particles.push_back(Particle{i, p_x, p_y, p_theta, 1});
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
	normal_distribution<double> x_pred_dist(0, std_pos[0]);
	normal_distribution<double> y_pred_dist(0, std_pos[1]);
	normal_distribution<double> theta_pred_dist(0, std_pos[2]);
	
	for (int i = 0; i < num_particles; ++i)
	{	
    	//predicted state values
    	double px_p = particles[i].x;
		double py_p = particles[i].y;
		double theta = particles[i].theta;

    	//avoid division by zero
    	if (fabs(yaw_rate) > 0.001) {
        	px_p += velocity/yaw_rate * ( sin (theta + yaw_rate*delta_t) - sin(theta));
        	py_p += velocity/yaw_rate * ( cos(theta) - cos(theta + yaw_rate*delta_t) );
    	}
    	else {
        	px_p += velocity*delta_t*cos(theta);
        	py_p += velocity*delta_t*sin(theta);
    	}

        double yaw_p = theta + yaw_rate*delta_t;

        //add noise
        //px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        //py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        //yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;		

		px_p += x_pred_dist(gen);
		py_p += y_pred_dist(gen);
		yaw_p += theta_pred_dist(gen);
		
		// Ensure that yaw is within 0 and 2 PI
		// Based on https://stackoverflow.com/questions/11498169/dealing-with-angle-wrap-in-c-code
        yaw_p = fmod(yaw_p + M_PI, 2.0 * M_PI);
        if (yaw_p < 0)
            yaw_p += 2.0 * M_PI;
        yaw_p -= M_PI;

		particles[i].x = px_p;
		particles[i].y = py_p;
		particles[i].theta = yaw_p;
	}
	
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    vector<LandmarkObs> associated_observations;
    for (int i = 0; i < predicted.size(); ++i)
    {
        double min_dist = 1e6;
        double min_index = -1;
        for (int j = 0; j < observations.size(); ++j)
        {
            double d = dist(predicted[i].x, predicted[i].y,
            				observations[j].x, observations[j].y);
            if (d < min_dist)
            {
            	min_dist = d;
            	min_index = j;
            }
        }
        
        // TODO: this is inefficient - replace with a list or consider Hungarian
        auto min_landmark = observations.erase(observations.begin() + min_index);
        associated_observations.push_back(*min_landmark);
    }
    
    observations.swap(associated_observations);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	for (int i = 0; i < num_particles; ++i)
	{
	    // transform the landmark within range into vehicle's coordinate frame
	    vector<LandmarkObs> predicted;
		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j)
		{
		    int id = map_landmarks.landmark_list[j].id_i;
		    float landmark_x = map_landmarks.landmark_list[j].x_f;
		    float landmark_y = map_landmarks.landmark_list[j].y_f;
		    double particle_x = particles[i].x;
		    double particle_y = particles[i].y;
		    double theta = particles[i].theta;
		    if (dist(landmark_x, landmark_y, particle_x, particle_y) < sensor_range)
		    {
		        double x = landmark_x - particle_x;
		        double y = landmark_y - particle_y;
		        double predicted_x = x*cos(theta) + y*sin(theta);
		        double predicted_y = x*sin(theta) + y*cos(theta);
		        predicted.push_back(LandmarkObs{id, predicted_x, predicted_y});
		    }
		}

        // associate predictions with observations
        dataAssociation(predicted, observations);
        
        // update particle weight using multivariate gaussian
        double new_weight = 0.0;
        for (int j = 0; j < predicted.size(); ++j)
        {
            double sq_x_diff = pow(predicted[j].x - observations[j].x, 2);
            double sq_y_diff = pow(predicted[j].y - observations[j].y, 2);
        	new_weight += 1/(2*M_PI*std_x*std_y)*exp(-sq_x_diff/(2*std_x*std_x) - sq_y_diff/(2*std_y*std_y));
        }
        
        particles[i].weight = new_weight;
        
	}
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	
	weights.clear();
	for (int i = 0; i < num_particles; ++i)
	{
		weights.push_back(particles[i].weight);
	}
	
	default_random_engine gen;
    discrete_distribution<> distribution(weights.begin(), weights.end());
	
	vector<Particle> new_particles;
	for (int i = 0; i < num_particles; ++i)
	{
	    int weighted_index = distribution(gen);
	    new_particles.push_back(particles[weighted_index]);
	}
	
	particles.swap(new_particles);
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
