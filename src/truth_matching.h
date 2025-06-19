
//
// Author: Gino Daniels (gino.christian.daniels@cern.ch)
//

#pragma once

#include <cmath>
#include <vector>
#include <math.h>

#include "TLorentzVector.h"
#include "ROOT/RVec.hxx"
#include "edm4hep/ReconstructedParticleData.h"
#include "edm4hep/MCParticleData.h"
#include "edm4hep/ParticleIDData.h"
#include "ReconstructedParticle2MC.h"


namespace FCCAnalyses {


// Takes in collection of mc tlv and reco tlv and a delta R threshold and returns a vector of indices of the reco tlv that are matched to the mc tlv
// If no match is found, the index is -1
struct TruthMatching {

static ROOT::VecOps::RVec<int> match_leptons(
    const ROOT::VecOps::RVec<TLorentzVector>& mc_tlvs,
    const ROOT::VecOps::RVec<TLorentzVector>& reco_tlvs,
    float deltaR_threshold) {

    ROOT::VecOps::RVec<int> reco_indices(mc_tlvs.size(), -1);
    std::vector<bool> used(reco_tlvs.size(), false);
    for (size_t i = 0; i < mc_tlvs.size(); ++i) {

        float min_dR = deltaR_threshold;
        int best_reco_idx = -1;

        for (size_t j = 0; j < reco_tlvs.size(); ++j) {

            if (used[j])
                continue;

            float dR = mc_tlvs[i].DeltaR(reco_tlvs[j]);

            if (dR < min_dR) {
                min_dR = dR;
                best_reco_idx = j;
            }
        }

        if (best_reco_idx >= 0) {
            reco_indices[i] = best_reco_idx;
            used[best_reco_idx] = true;
        }
    }

    return reco_indices;
}


// Delta R calc between two tlv collections
static ROOT::VecOps::RVec<float> Delta_R_calc(
    const ROOT::VecOps::RVec<TLorentzVector>& mc_tlvs,
    const ROOT::VecOps::RVec<TLorentzVector>& reco_tlvs) {

    ROOT::VecOps::RVec<float> delta_r_values;
    delta_r_values.reserve(mc_tlvs.size() * reco_tlvs.size());

    for (size_t i = 0; i < mc_tlvs.size(); ++i) {
        for (size_t j = 0; j < reco_tlvs.size(); ++j) {

            float dR = mc_tlvs[i].DeltaR(reco_tlvs[j]);
            delta_r_values.push_back(dR);
        }
    }
  
  return delta_r_values;
}


// Delta R calc between two tlv collections, returns the minimum delta R value for each lepton
static ROOT::VecOps::RVec<float> Delta_R_min_calc(
    const ROOT::VecOps::RVec<TLorentzVector>& tlvs_1_jets,
    const ROOT::VecOps::RVec<TLorentzVector>& tlvs_2_leptons) {

    ROOT::VecOps::RVec<float> delta_r_values;
    delta_r_values.reserve(tlvs_2_leptons.size());

    for (size_t i = 0; i < tlvs_2_leptons.size(); ++i) {

        float min_dR = std::numeric_limits<float>::max();

        for (size_t j = 0; j < tlvs_1_jets.size(); ++j) {

            float dR = tlvs_2_leptons[i].DeltaR(tlvs_1_jets[j]);
            if (dR < min_dR) {
                min_dR = dR;
            }
        }
    
        delta_r_values.push_back(min_dR);
    }

    return delta_r_values;
}


// Compute z0 for all particles from a collection of vertices and tlvs, was used to keep track of z0 vals for truth matched leptons then look at correlation between z0 and D Iso
static ROOT::VecOps::RVec<float> compute_z0(
    const ROOT::VecOps::RVec<edm4hep::Vector3d>& vertices,
    const ROOT::VecOps::RVec<TLorentzVector>& tlvs) {

    ROOT::VecOps::RVec<float> z0_values;
    z0_values.reserve(vertices.size());

    for (size_t i = 0; i < vertices.size(); ++i) {

        TVector3 x(vertices[i].x, vertices[i].y, vertices[i].z);
        TVector3 p(tlvs[i].Px(), tlvs[i].Py(), tlvs[i].Pz());
        float z0 = FCCAnalyses::myUtils::get_z0(x, p);
        z0_values.push_back(z0);
    }

    return z0_values;
}


static std::pair<ROOT::VecOps::RVec<int>, ROOT::VecOps::RVec<float>>
match_leptons_with_z0(
    const ROOT::VecOps::RVec<TLorentzVector>& mc_tlvs,
    const ROOT::VecOps::RVec<TLorentzVector>& reco_tlvs,
    const ROOT::VecOps::RVec<edm4hep::Vector3d>& mc_vertices,
    float deltaR_threshold) {

    ROOT::VecOps::RVec<int> reco_indices(mc_tlvs.size(), -1);
    ROOT::VecOps::RVec<float> z0_values;
    z0_values.reserve(mc_tlvs.size());
    std::vector<bool> used(reco_tlvs.size(), false);

    // Compute z0 for all MC leptons
    for (size_t i = 0; i < mc_tlvs.size() && i < mc_vertices.size(); ++i) {
        TVector3 x(mc_vertices[i].x, mc_vertices[i].y, mc_vertices[i].z);
        TVector3 p(mc_tlvs[i].Px(), mc_tlvs[i].Py(), mc_tlvs[i].Pz());
        float z0 = FCCAnalyses::myUtils::get_z0(x, p);
        z0_values.push_back(z0);
    }

    // // Fill with default 0.0 if vertices are fewer than tlvs
    // while (z0_values.size() < mc_tlvs.size()) {
    //     z0_values.push_back(-1);
    // }

    // Perform truth matching
    for (size_t i = 0; i < mc_tlvs.size(); ++i) {

        float min_dR = deltaR_threshold;
        int best_reco_idx = -1;
        for (size_t j = 0; j < reco_tlvs.size(); ++j) {

            if (used[j])
                continue;

            float dR = mc_tlvs[i].DeltaR(reco_tlvs[j]);
            if (dR < min_dR) {
                min_dR = dR;
                best_reco_idx = j;
            }
        }

        if (best_reco_idx >= 0) {
            reco_indices[i] = best_reco_idx;
            used[best_reco_idx] = true;
        }
    }

    return std::make_pair(reco_indices, z0_values);
}


static std::pair<ROOT::VecOps::RVec<int>, ROOT::VecOps::RVec<int>>
getJetMotherPdgId(
    const ROOT::VecOps::RVec<fastjet::PseudoJet>& jets,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData>& mcParticles,
    const ROOT::VecOps::RVec<int>& recIndices,
    const ROOT::VecOps::RVec<int>& mcIndices,
    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& recoParticles,
    const ROOT::VecOps::RVec<int>& Particle0, // Parent indices Particle0
    const ROOT::VecOps::RVec<int>& Particle1, // Daughter indices Particle1
    float max_angle = 0.3) {

    ROOT::VecOps::RVec<int> jetFlavors(jets.size(), 0);
    ROOT::VecOps::RVec<int> motherPdgIds(jets.size(), 0);

    for (size_t j = 0; j < jets.size(); ++j) {

        const auto& jet = jets[j];

        for (size_t i = 0; i < mcParticles.size(); ++i) {

            const auto& parton = mcParticles[i];
            // std::cout << " parton.generator  status " << parton.generatorStatus << std::endl;
            // Select partons only (for pythia8 71-79, for pythia6 2)
            if (
                (parton.generatorStatus > 80 || parton.generatorStatus < 70) &&
                parton.generatorStatus != 2) {
                continue;
            }

            if (std::abs(parton.PDG) > 5 && parton.PDG != 21) {
                continue;
            }


            Float_t dot = jet.px() * parton.momentum.x + jet.py() * parton.momentum.y +
            jet.pz() * parton.momentum.z;
            Float_t lenSq1 = jet.px() * jet.px() + jet.py() * jet.py() + jet.pz() * jet.pz();
            Float_t lenSq2 = parton.momentum.x * parton.momentum.x +
            parton.momentum.y * parton.momentum.y +
            parton.momentum.z * parton.momentum.z;
            Float_t norm = sqrt(lenSq1 * lenSq2);
            Float_t angle = acos(dot / norm);

            if (angle <= max_angle) {
                int current_pdg = std::abs(parton.PDG);
                bool update_mother = false;

                // If the jet flavor is 5 or 21 go to next mother to get the pdgid 
                if (jetFlavors[j] == 0 || jetFlavors[j] == 21) {

                    jetFlavors[j] = current_pdg;
                    update_mother = true;

                } else if (parton.PDG != 21) {

                    int new_flavor = std::max(jetFlavors[j], current_pdg);
                    if (new_flavor > jetFlavors[j]) {
                        jetFlavors[j] = new_flavor;
                        update_mother = true;
                    }
                }

                if (update_mother && parton.parents_begin < parton.parents_end &&
                    parton.parents_begin < Particle0.size()) {

                    int selected_mother_pdg = 0;
                    // Check immediate parents to find the first one that is not the same as the jet flavor
                    for (unsigned int p = parton.parents_begin; p < parton.parents_end && p < Particle0.size(); ++p) {
                        
                        int parent_idx = Particle0[p];
                        if (parent_idx < 0 || parent_idx >= mcParticles.size())
                            continue;
                        
                        int parent_pdg = std::abs(mcParticles[parent_idx].PDG);
                        if(parent_pdg == abs(6)){
                            selected_mother_pdg = parent_pdg;
                            break;
                        }

                        // Select the first parent that is not the same as the jet flavor
                        if (parent_pdg != jetFlavors[j]) {
                            selected_mother_pdg = parent_pdg;
                            break; // Use the first parent that is different from the jet flavor
                        }
                    }

                    // Only update if a valid mother was found
                    if (selected_mother_pdg != 0) {
                        motherPdgIds[j] = selected_mother_pdg;
                    }
                }
            }
        }
    }

    return std::make_pair(jetFlavors, motherPdgIds);
}


static int get_particle_origin(
    const edm4hep::MCParticleData &p,
    const ROOT::VecOps::RVec<edm4hep::MCParticleData> &in,
    const ROOT::VecOps::RVec<int> &ind) {

    // std::cout  << std::endl << " enter in MCParticle::get_lepton_origin  PDG = " << p.PDG << std::endl;

    int result = 0;

    //  std::cout << " p.parents_begin p.parents_end " << p.parents_begin <<  " "  << p.parents_end << std::endl;
    for (unsigned j = (p.parents_begin); j <(p.parents_end); ++j) {

        if (j >= ind.size())
            continue; 

        int index = ind.at(j);

        if (index < 0 || index >= in.size())
            continue; // Safety check for valid index

        int pdg_parent = in.at(index).PDG ;

        // std::cout << " pdg_parent " << pdg_parent << std::endl;
        if(pdg_parent == 6)
            result = pdg_parent;

        if (pdg_parent== 11 || pdg_parent== -11)
            continue;

        else (result=pdg_parent); //I dont understand why this doesnt ever show a top quark and only the 11,-11
            // std::cout  << " parent has pdg = " << in.at(index).PDG <<  "  status = " << in.at(index).generatorStatus << std::endl;
        break;
        // result = pdg_parent;
        // // std::cout <<  " ... Parent PDG ID found, return code = " << result <<  std::endl;
        // break; // Return the first parent's PDG ID
    }

    return result;
}


static ROOT::VecOps::RVec<int>
get_particles_origin(const ROOT::VecOps::RVec<edm4hep::MCParticleData> &particles,
                     const ROOT::VecOps::RVec<edm4hep::MCParticleData> &in,
                     const ROOT::VecOps::RVec<int> &ind)  {

    ROOT::VecOps::RVec<int> result;
    result.reserve(particles.size());
    for (size_t i = 0; i < particles.size(); ++i) {
        auto & p = particles[i];
        int origin = get_particle_origin(p, in, ind);
        result.push_back(origin);
    }

    return result;
}


static ROOT::VecOps::RVec<bool> computeJetRemovalMask(
    const ROOT::VecOps::RVec<TLorentzVector>& jets,
    const ROOT::VecOps::RVec<TLorentzVector>& muons,
    const ROOT::VecOps::RVec<TLorentzVector>& electrons,
    float deltaR_threshold) {

    ROOT::VecOps::RVec<bool> mask(jets.size(), true);
    for (size_t j = 0; j < jets.size(); ++j) {

        for (size_t i = 0; i < muons.size(); ++i) {
            if (jets[j].DeltaR(muons[i]) < deltaR_threshold) {
                mask[j] = false;
                break; // Jet is already excluded, no need to check further
            }
        }

        if (mask[j]) { // Only check electrons if jet wasn’t excluded by muons
            for (size_t i = 0; i < electrons.size(); ++i) {
                if (jets[j].DeltaR(electrons[i]) < deltaR_threshold) {
                    mask[j] = false;
                    break;
                }
            }
        }
    }

    return mask;
}

};

} // namespace FCCAnalyses
