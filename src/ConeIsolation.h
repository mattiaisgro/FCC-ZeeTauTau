
// Originally from the ttThreshold examples

#pragma once

#include <vector>
#include "Math/Vector4D.h"
#include "TLorentzVector.h"
#include "ROOT/RVec.hxx"


namespace FCCAnalyses::ConeIsolation {

	// Selects particles with isolation less than max_iso.
	struct sel_iso {
		float m_max_iso = 0.25f;
		sel_iso(float arg_max_iso);
		Vec_rp operator()(Vec_rp in, Vec_f iso);
	};

	inline sel_iso::sel_iso(float arg_max_iso) : m_max_iso(arg_max_iso) {}

	inline Vec_rp sel_iso::operator()(Vec_rp in, Vec_f iso) {

		Vec_rp result;
		result.reserve(in.size());
		for (size_t i = 0; i < in.size(); ++i) {
			if (iso[i] < m_max_iso) {
				result.emplace_back(in[i]);
			}
		}
		
		return result;
	}


	// Computes the cone isolation for reconstructed particles.
	struct coneIsolation {
		float dr_min = 0.0f;
		float dr_max = 0.4f;

		coneIsolation(float arg_dr_min, float arg_dr_max);

		static double deltaR(double eta1, double phi1, double eta2, double phi2) {
			return std::sqrt(std::pow(eta1 - eta2, 2) + std::pow(phi1 - phi2, 2));
		}

		Vec_f operator()(Vec_rp in, Vec_rp rps);
	};

	inline coneIsolation::coneIsolation(float arg_dr_min, float arg_dr_max)
		: dr_min(arg_dr_min), dr_max(arg_dr_max) {}

	inline Vec_f coneIsolation::operator()(Vec_rp in, Vec_rp rps) {
		Vec_f result;
		result.reserve(in.size());

		std::vector<ROOT::Math::PxPyPzEVector> lv_reco, lv_charged, lv_neutral;

		for (const auto& rp : rps) {
			ROOT::Math::PxPyPzEVector tlv;
			tlv.SetPxPyPzE(rp.momentum.x, rp.momentum.y, rp.momentum.z, rp.energy);
			if (rp.charge == 0)
				lv_neutral.push_back(tlv);
			else
				lv_charged.push_back(tlv);
		}

		for (const auto& rp : in) {
			ROOT::Math::PxPyPzEVector tlv;
			tlv.SetPxPyPzE(rp.momentum.x, rp.momentum.y, rp.momentum.z, rp.energy);
			lv_reco.push_back(tlv);
		}

		for (const auto& lv_reco_ : lv_reco) {
			double sumNeutral = 0.0, sumCharged = 0.0;

			for (const auto& lv_charged_ : lv_charged) {
				double dr = deltaR(lv_reco_.Eta(), lv_reco_.Phi(), lv_charged_.Eta(), lv_charged_.Phi());
				if (dr > dr_min && dr < dr_max)
					sumCharged += lv_charged_.P();
			}

			for (const auto& lv_neutral_ : lv_neutral) {
				double dr = deltaR(lv_reco_.Eta(), lv_reco_.Phi(), lv_neutral_.Eta(), lv_neutral_.Phi());
				if (dr > dr_min && dr < dr_max)
					sumNeutral += lv_neutral_.P();
			}

			double sum = sumCharged + sumNeutral;
			double ratio = sum / lv_reco_.P();
			result.emplace_back(ratio);
		}

		return result;
	}

} // namespace FCCAnalyses::ConeIsolation
