/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <cassert>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    num_particles = 5;

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

    random_device rd;
    default_random_engine gen(rd());
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

        theta += yaw_rate*delta_t;

        //add noise
        px_p += x_pred_dist(gen);
        py_p += y_pred_dist(gen);
        theta += theta_pred_dist(gen);
        
        // Ensure that yaw is within 0 and 2 PI
        // Based on https://stackoverflow.com/questions/11498169/dealing-with-angle-wrap-in-c-code
        theta = fmod(theta + M_PI, 2.0 * M_PI);
        if (theta < 0)
            theta += 2.0 * M_PI;
        theta -= M_PI;

        particles[i].x = px_p;
        particles[i].y = py_p;
        particles[i].theta = theta;

    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

    for (int i = 0; i < observations.size(); ++i)
    {
        double min_dist = INFINITY;
        for (int j = 0; j < predicted.size(); ++j)
        { 
            double d = dist(predicted[j].x, predicted[j].y, 
                            observations[i].x, observations[i].y);
            if (d < min_dist)
            {
                min_dist = d;
                observations[i].id = j;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        std::vector<LandmarkObs> observations, Map map_landmarks) {

    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    for (int i = 0; i < num_particles; ++i)
    {
        double particle_x = particles[i].x;
        double particle_y = particles[i].y;
        double particle_theta = particles[i].theta;

        // gather all landmarks within sensor range
        vector<LandmarkObs> predicted_landmarks;
        for (int j = 0; j < map_landmarks.landmark_list.size(); ++j)
        {
            int id = map_landmarks.landmark_list[j].id_i;
            float landmark_x = map_landmarks.landmark_list[j].x_f;
            float landmark_y = map_landmarks.landmark_list[j].y_f;
            if (dist(landmark_x, landmark_y, particle_x, particle_y) < sensor_range)
            {
                predicted_landmarks.push_back(LandmarkObs{id, landmark_x, landmark_y});
            }
        }

        // transform observations to map coordinate frame
        vector<LandmarkObs> transformed_observations;
        for (int j = 0; j < observations.size(); ++j)
        {
            double x = observations[j].x;
            double y = observations[j].y;
            double transformed_x = x*cos(particle_theta) - y*sin(particle_theta);
            double transformed_y = x*sin(particle_theta) + y*cos(particle_theta);
            transformed_x += particle_x;
            transformed_y += particle_y;
            transformed_observations.push_back(LandmarkObs{j, transformed_x, transformed_y});
        }

        // associate predictions with observations using nearest neighbour
        dataAssociation(predicted_landmarks, transformed_observations);

        // update particle weight using multivariate gaussian
        double scale = 0.5 / (M_PI * std_x * std_y);
        double new_weight = 0.0;
        for (auto obs: transformed_observations)
        {
            cout << obs.id << " " << obs.x << " " << obs.y << endl; 
            cout << " obs is associated with landmark" << obs.id << endl;
            LandmarkObs pred = predicted_landmarks[obs.id];
            double x_diff = pow(obs.x - pred.x, 2);
            double y_diff = pow(obs.y - pred.y, 2);
            new_weight += scale * exp(-x_diff/(2*std_x*std_x) - y_diff/(2*std_y*std_y));
        }
        cout << " reweighting particle " << i << " from " << particles[i].weight << " to " << new_weight << endl; 
        particles[i].weight = new_weight;
        
    }
}

void ParticleFilter::resample() {

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
        cout << " drawing particle " << weighted_index << endl; 
        new_particles.push_back(particles[weighted_index]);
    }

    particles.swap(new_particles);
}

void ParticleFilter::write(std::string filename) {
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << " " << particles[i].weight << "\n";
    }
    dataFile.close();
}
