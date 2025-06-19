
//
// Author: Mattia Isgrò (mattia.isgro@cern.ch)
//

#ifndef ZTAUTAUFUNCTION_H
#define ZTAUTAUFUNCTION_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

#include "TLorentzVector.h"
#include "ROOT/RVec.hxx"
#include "edm4hep/ReconstructedParticleData.h"
#include "edm4hep/MCParticleData.h"
#include "edm4hep/ParticleIDData.h"
#include "ReconstructedParticle2MC.h"
#include "fastjet/JetDefinition.hh"
#include "Math/Vector4D.h"
#include "TRandom3.h"


namespace FCCAnalyses::ZTauTau {


	// Match tau jets to MC particles, making sure to select only opposite charge taus.
	// The sign of the particle code is preserved and for jets which are not tau,
	// the PDG code is set to 0.
	ROOT::VecOps::RVec<int>
	match_tau_flavor(
		ROOT::VecOps::RVec<fastjet::PseudoJet> jets_in,
		ROOT::VecOps::RVec<edm4hep::MCParticleData> MCin) {

		ROOT::VecOps::RVec<int> result(jets_in.size(), 0);

		for (size_t i = 0; i < MCin.size(); ++i) {

			auto &p_mc = MCin[i];

			// Select only taus
			if (std::abs(p_mc.PDG) != 15)
				continue;

			for (size_t j = 0; j < jets_in.size(); ++j) {

				auto &p = jets_in[j];

				Float_t dot =
					p.px() * p_mc.momentum.x + p.py() * p_mc.momentum.y + p.pz() * p_mc.momentum.z;
				
				Float_t lenSq1 =
					p.px() * p.px() + p.py() * p.py() + p.pz() * p.pz();

				Float_t lenSq2 = 
					p_mc.momentum.x * p_mc.momentum.x +
					p_mc.momentum.y * p_mc.momentum.y +
					p_mc.momentum.z * p_mc.momentum.z;

				Float_t norm = sqrt(lenSq1 * lenSq2);
				Float_t angle = acos(dot / norm);
				const Float_t angle_tolerance = 0.5;

				// Match taus to jets by angle
				if (angle <= angle_tolerance) {
					result[j] = p_mc.PDG;
				}
			}
		}

		return result;
	}


	// Get tau-tags with an efficiency applied.
	ROOT::VecOps::RVec<int> get_tautag(
		ROOT::VecOps::RVec<int> jets_in, float efficiency) {

		ROOT::VecOps::RVec<int> result(jets_in.size(), 0);

		for (size_t j = 0; j < jets_in.size(); ++j) {

			if (std::abs(jets_in.at(j)) == 15 && gRandom->Uniform() <= efficiency)
				result[j] = 1;
		}

		return result;
	}


	// Select MC leptons resulting from a Z boson decay.
	ROOT::VecOps::RVec<edm4hep::MCParticleData> sel_genlepsfromZ(
		const ROOT::VecOps::RVec<edm4hep::MCParticleData>& leptons,
		const ROOT::VecOps::RVec<int>& mother_pdgIds) {

		ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
		result.reserve(leptons.size());

		for (size_t i = 0; i < leptons.size(); ++i) {

			if (std::abs(mother_pdgIds[i]) == 23) {
				result.emplace_back(leptons[i]);
			}
		}

		return result;
	}
}

#endif
