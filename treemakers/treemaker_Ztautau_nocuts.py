
#
# Analysis of Z -> tau tau events for 3-prong decays
#
# Test command:
# fccanalysis run treemaker_Ztautau_reconstruction.py

from argparse import ArgumentParser
import copy
import os
import urllib
import getpass

# Jet flavour tagging and clustering helpers
from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import InclusiveJetClusteringHelper


global jetClusteringHelper
global jetFlavourHelper


# Get a file from a URL or local path.
def get_file_path(url, filename):

    if os.path.exists(filename):
        return os.path.abspath(filename)
    else:
        urllib.request.urlretrieve(url, os.path.basename(url))
        return os.path.basename(url)


# Get a list of files in a directory on EOS.
def get_files(eos_dir, proc):

    files=[]
    basepath=os.path.join(eos_dir, proc)
    if os.path.exists(basepath):
        files =  [
            os.path.join(basepath,x) for x in os.listdir(basepath) if os.path.isfile(os.path.join(basepath, x))
        ]

    return files


# Load a jet-tagging model and its preprocessing file from the Winter 2023 dataset.
def load_jet_model(model_name):

    model_dir = "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"

    local_preproc = "{}/{}.json".format(model_dir, model_name)
    local_model = "{}/{}.onnx".format(model_dir, model_name)

    url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
    url_preproc = "{}/{}.json".format(url_model_dir, model_name)
    url_model = "{}/{}.onnx".format(url_model_dir, model_name)
    
    return get_file_path(url_preproc, local_preproc), get_file_path(url_model, local_model)


# Clean jets by removing those overlapping with leptons
def clean_jets(df, jetClusteringHelper, deltaR_threshold=0.5):
    
    df = df.Define("jets_p_unfiltered", "JetClusteringUtils::get_p({})".format(jetClusteringHelper.jets))
    df = df.Define("jets_p_unfiltered_size", "jets_p_unfiltered.size()")
    df = df.Define("jets_theta_unfiltered", "JetClusteringUtils::get_theta({})".format(jetClusteringHelper.jets))
    df = df.Define("jets_phi_unfiltered", "JetClusteringUtils::get_phi({})".format(jetClusteringHelper.jets))
    df = df.Define("jets_tlv_unfiltered", "JetConstituentsUtils::compute_tlv_jets({})".format(jetClusteringHelper.jets))

    df = df.Define("remaining_muons_tlv", "FCCAnalyses::ReconstructedParticle::get_tlv(muons_sel_iso)")
    df = df.Define("remaining_electrons_tlv", "FCCAnalyses::ReconstructedParticle::get_tlv(electrons_sel_iso)")

    df = df.Define("remaining_muons_deltaR", "FCCAnalyses::TruthMatching::Delta_R_calc(jets_tlv_unfiltered, remaining_muons_tlv)")
    df = df.Define("remaining_electrons_deltaR", "FCCAnalyses::TruthMatching::Delta_R_calc(jets_tlv_unfiltered, remaining_electrons_tlv)")
    
    df = df.Define("jet_removal_mask",
                f"""
                ROOT::VecOps::RVec<bool> mask(jets_theta_unfiltered.size(), true);
                // Check Delta R for muons
                for(size_t i = 0; i < remaining_muons_tlv.size(); i++) {{
                    for(size_t j = 0; j < jets_theta_unfiltered.size(); j++) {{
                        size_t deltaR_index = i * jets_theta_unfiltered.size() + j;
                        if(deltaR_index < remaining_muons_deltaR.size() && remaining_muons_deltaR[deltaR_index] < {deltaR_threshold}) {{
                            mask[j] = false;
                        }}
                    }}
                }}
                // Check Delta R for electrons
                for(size_t i = 0; i < remaining_electrons_tlv.size(); i++) {{
                    for(size_t j = 0; j < jets_theta_unfiltered.size(); j++) {{
                        size_t deltaR_index = i * jets_theta_unfiltered.size() + j;
                        if(deltaR_index < remaining_electrons_deltaR.size() && remaining_electrons_deltaR[deltaR_index] < {deltaR_threshold}) {{
                            mask[j] = false;
                        }}
                    }}
                }}
                return mask;
                """
    )

    # Apply the mask to filter out jets that are too close to leptons
    df = df.Define("jets_tlv", "jets_tlv_unfiltered[jet_removal_mask]")
    df = df.Define("jets_p", "jets_p_unfiltered[jet_removal_mask]")
    df = df.Define("jets_p_size", "jets_p.size()")
    df = df.Define("jets_theta", "jets_theta_unfiltered[jet_removal_mask]")
    df = df.Define("jets_phi", "jets_phi_unfiltered[jet_removal_mask]")
    
    return df


# Center of Mass Energy
ecm       = 91
print("Center of Mass Energy = " + str(ecm))


# Channel to be reconstructed ("leptonic", "semihadronic", "hadronic")
channel = "hadronic"

if channel not in ["leptonic", "semihadronic", "hadronic"]:
    print("Using default channel settings...")
    channel = "hadronic"

print("Channel = " + channel)

# latest particle transformer model, trained on 9M jets in winter2023 samples
weaverPreproc , weaverModel = load_jet_model("fccee_flavtagging_edm4hep_wc")

jetFlavourHelper = None
jetClusteringHelper = None


# Analysis class used to apply the analysis and pick the outputs
class Analysis:

    # Initialize the analysis class with run parameters
    def __init__(self, cmdline_args):
        
        # Parse command line arguments
        parser = ArgumentParser(
            description='Additional analysis arguments',
            usage='Provide additional arguments after analysis script path'
        )
        self.ana_args, _ = parser.parse_known_args(cmdline_args['unknown'])

        # List of processes
        self.process_list = {
            'p8_ee_Ztautau_ecm91': {
                'fraction': 1,
                'chunks': 1000,
            },
        }

        # Input directory where to find the samples
        self.input_dir = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/"

        # Optional: output directory, default is local running directory
        username = getpass.getuser()
        self.output_dir = f"/eos/user/{username[0]}/{username}/Ztautau/reco_nocuts/{channel}"

        # Title of the analysis
        self.analysis_name = 'Analysis and reconstruction of Z -> tau tau events'

        # Number of threads to run on
        self.n_threads = 4

        # Run on Condor
        self.run_batch = True

        # Whether to use weighted events
        self.do_weighted = True

        # Whether to read the input files with podio::DataSource
        self.use_data_source = False


    # Run the analysis and return the resulting DataFrame
    def analyzers(self, df):

        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Electron0","Electron#0.index")
        df = df.Alias("Particle0","Particle#0.index")


        # Get all the leptons from the collection
        df = df.Define(
            "muons_all",
            "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)",
        )

        df = df.Define(
            "electrons_all",
            "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)",
        )


        # Remove unprompted leptons with a momentum threshold
        leptonMomentumThreshold = 0.0  # GeV

        df = df.Define(
            "muons_sel",
            "FCCAnalyses::ReconstructedParticle::sel_p({})(muons_all)".format(leptonMomentumThreshold),
        )

        df = df.Define(
            "electrons_sel",
            "FCCAnalyses::ReconstructedParticle::sel_p({})(electrons_all)".format(leptonMomentumThreshold),
        )


        # Select muons and electrons with an isolation cut of 0df = df.25 in a separate column
        isolationThreshold = 0.30

        df = df.Define(
            "muons_isolation",
            "FCCAnalyses::ConeIsolation::coneIsolation(0.01, 0.3)(muons_sel, ReconstructedParticles)",
        )

        df = df.Define(
            "electrons_isolation",
            "FCCAnalyses::ConeIsolation::coneIsolation(0.01, 0.3)(electrons_sel, ReconstructedParticles)",
        )

        df = df.Define(
            "muons_sel_iso",
            "FCCAnalyses::ConeIsolation::sel_iso({})(muons_sel, muons_isolation)".format(isolationThreshold),
        )

        df = df.Define(
            "electrons_sel_iso",
            "FCCAnalyses::ConeIsolation::sel_iso({})(electrons_sel, electrons_isolation)".format(isolationThreshold),
        )


        # Select events based on the number of isolated muons and electrons
        #print("FILTER: Filtering events in channel '{}'".format(channel))
        #df = remove_iso_leptons(df, channel)


        # If the channel is not hadronic, define the kinematic variables for muons and electrons
        if channel != "hadronic":

            df = df.Define(
                "muons_p", "FCCAnalyses::ReconstructedParticle::get_p(muons_sel_iso)"
            )

            df = df.Define(
                "electrons_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons_sel_iso)"
            )

            df = df.Define(
                "muons_theta",
                "FCCAnalyses::ReconstructedParticle::get_theta(muons_sel_iso)",
            )

            df = df.Define(
                "muons_phi",
                "FCCAnalyses::ReconstructedParticle::get_phi(muons_sel_iso)",
            )

            df = df.Define(
                "muons_q",
                "FCCAnalyses::ReconstructedParticle::get_charge(muons_sel_iso)",
            )

            df = df.Define(
                "muons_n", "FCCAnalyses::ReconstructedParticle::get_n(muons_sel_iso)",
            )

            df = df.Define(
                "electrons_theta",
                "FCCAnalyses::ReconstructedParticle::get_theta(electrons_sel_iso)",
            )

            df = df.Define(
                "electrons_phi",
                "FCCAnalyses::ReconstructedParticle::get_phi(electrons_sel_iso)",
            )
            
            df = df.Define(
                "electrons_q",
                "FCCAnalyses::ReconstructedParticle::get_charge(electrons_sel_iso)",
            )
            
            df = df.Define(
                "electrons_n", "FCCAnalyses::ReconstructedParticle::get_n(electrons_sel_iso)",
            )

        
        # Cluster jets in the events, but first remove muons from the list of reconstructed particles
        # Create a new collection of reconstructed particles removing muons and electrons with p > 12 GeV
        df = df.Define(
            "ReconstructedParticlesNoMuons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, muons_sel_iso)",
        )
        
        df = df.Define(
            "ReconstructedParticlesNoMuNoEl",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoMuons, electrons_sel_iso)",
        )


        global jetClusteringHelper
        global jetFlavourHelper
        
        # Define jet and run clustering parameters (names of collections in EDM root files)
        collections = {
            "GenParticles": "Particle",
            "PFParticles": "ReconstructedParticles",
            "PFTracks": "EFlowTrack",
            "PFPhotons": "EFlowPhoton",
            "PFNeutralHadrons": "EFlowNeutralHadron",
            "TrackState": "EFlowTrack_1",
            "TrackerHits": "TrackerHits",
            "CalorimeterHits": "CalorimeterHits",
            "dNdx": "EFlowTrack_2",
            "PathLength": "EFlowTrack_L",
            "Bz": "magFieldBz",
        }

        collections_noleps = copy.deepcopy(collections)
        collections_noleps["PFParticles"] = "ReconstructedParticlesNoMuNoEl"
        
        jetMomentumThreshold = 3.0 # GeV
        jetClusteringHelper  = InclusiveJetClusteringHelper(
            collections_noleps["PFParticles"], 0.5, jetMomentumThreshold, "R5"
        )

        df = jetClusteringHelper.define(df)

        jetFlavourHelper = JetFlavourHelper(
            collections_noleps,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
            "R5",
        )
        
        df = jetFlavourHelper.define(df)
        df = jetFlavourHelper.inference(weaverPreproc, weaverModel, df)


        # Define lepton kinematic variables
        df = df.Define(
            "lep_p",
            "muons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_p(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_p(electrons_sel_iso)[0] : -999) "
        )

        df = df.Define(
            'lep_theta',
            'muons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_theta(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_theta(electrons_sel_iso)[0] : -999) '
        )

        df = df.Define(
            'lep_phi',
            'muons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_phi(muons_sel_iso)[0] : (electrons_sel_iso.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_phi(electrons_sel_iso)[0] : -999) '
        )

        df = df.Define(
            "nlep",
            "electrons_sel_iso.size() + muons_sel_iso.size()"
        )

        df = df.Define(
            "missing_p",
            "FCCAnalyses::ReconstructedParticle::get_p(MissingET)[0]",
        )
        
        df = df.Define(
            'missing_p_theta', 'ReconstructedParticle::get_theta(MissingET)[0]',
        )

        df = df.Define(
            'missing_p_phi', 'ReconstructedParticle::get_phi(MissingET)[0]',
        )


        # Clean jets to remove those overlapping with leptons
        #print("FILTER: Cleaning jets overlapping with leptons (deltaR_threshold = {})...".format(0.4))
        #df = clean_jets(df, jetClusteringHelper, deltaR_threshold=0.4)
        # deltaR threshold of 0.4, following ATLAS
        
        df = df.Define("jets_total_mass", "FCCAnalyses::ZTauTau::get_jets_total_m({})".format(jetClusteringHelper.jets))
        df = df.Define("nprongs", "FCCAnalyses::ZTauTau::get_nprongs({})".format(jetClusteringHelper.constituents))

        # Kinematic variables of jets (UNFILTERED)
        df = df.Define("jets_p", "JetClusteringUtils::get_p({})".format(jetClusteringHelper.jets))
        df = df.Define("jets_theta", "JetClusteringUtils::get_theta({})".format(jetClusteringHelper.jets))
        df = df.Define("jets_phi", "JetClusteringUtils::get_phi({})".format(jetClusteringHelper.jets))

        # Define number of jets
        df = df.Define("njets", "jets_p.size()")

        df = df.Define("jet1_p", "jets_p.size() > 0 ? jets_p[0] : -999")
        df = df.Define("jet2_p", "jets_p.size() > 1 ? jets_p[1] : -999")
        df = df.Define("jet3_p", "jets_p.size() > 2 ? jets_p[2] : -999")
        df = df.Define("jet4_p", "jets_p.size() > 3 ? jets_p[3] : -999")
        df = df.Define("jet5_p", "jets_p.size() > 4 ? jets_p[4] : -999")

        df = df.Define("jet1_theta", "jets_theta.size() > 0 ? jets_theta[0] : -999")
        df = df.Define("jet2_theta", "jets_theta.size() > 1 ? jets_theta[1] : -999")
        df = df.Define("jet3_theta", "jets_theta.size() > 2 ? jets_theta[2] : -999")
        df = df.Define("jet4_theta", "jets_theta.size() > 3 ? jets_theta[3] : -999")
        df = df.Define("jet5_theta", "jets_theta.size() > 4 ? jets_theta[4] : -999")

        df = df.Define("jet1_phi", "jets_phi.size() > 0 ? jets_phi[0] : -999")
        df = df.Define("jet2_phi", "jets_phi.size() > 1 ? jets_phi[1] : -999")
        df = df.Define("jet3_phi", "jets_phi.size() > 2 ? jets_phi[2] : -999")
        df = df.Define("jet4_phi", "jets_phi.size() > 3 ? jets_phi[3] : -999")
        df = df.Define("jet5_phi", "jets_phi.size() > 4 ? jets_phi[4] : -999")

        df = df.Define("jets_delta_phi", "std::abs(jet1_phi - jet2_phi)")
        df = df.Define("jets_delta_theta", "std::abs(jet1_theta - jet2_theta)")
        df = df.Define("jets_delta_r", "std::sqrt(jets_delta_phi * jets_delta_phi + jets_delta_theta * jets_delta_theta)")

        # Tau jet score between 0 and 1
        df = df.Define("jet1_tau_score", "jets_phi.size() > 0 ? recojet_isTAU_R5[0] : -999")
        df = df.Define("jet2_tau_score", "jets_phi.size() > 1 ? recojet_isTAU_R5[1] : -999")
        df = df.Define("jet3_tau_score", "jets_phi.size() > 2 ? recojet_isTAU_R5[2] : -999")
        df = df.Define("jet4_tau_score", "jets_phi.size() > 3 ? recojet_isTAU_R5[3] : -999")
        df = df.Define("jet5_tau_score", "jets_phi.size() > 4 ? recojet_isTAU_R5[4] : -999")

        # Compute the invariant masses of the jets
        df = df.Define("jets_mass", "FCCAnalyses::JetClusteringUtils::get_m({})".format(jetClusteringHelper.jets))
        df = df.Define("jet1_mass", "jets_phi.size() > 0 ? jets_mass[0] : -999")
        df = df.Define("jet2_mass", "jets_phi.size() > 1 ? jets_mass[1] : -999")
        df = df.Define("jet3_mass", "jets_phi.size() > 2 ? jets_mass[2] : -999")
        df = df.Define("jet4_mass", "jets_phi.size() > 3 ? jets_mass[3] : -999")
        df = df.Define("jet5_mass", "jets_phi.size() > 4 ? jets_mass[4] : -999")

        df = df.Define(
            "jet1_total_charge",
            "FCCAnalyses::ZTauTau::get_jet_total_charge({}, 0)".format(jetClusteringHelper.constituents)
        )

        df = df.Define(
            "jet2_total_charge",
            "FCCAnalyses::ZTauTau::get_jet_total_charge({}, 1)".format(jetClusteringHelper.constituents)
        )

        df = df.Define(
            "jet3_total_charge",
            "FCCAnalyses::ZTauTau::get_jet_total_charge({}, 2)".format(jetClusteringHelper.constituents)
        )

        df = df.Define(
            "jet4_total_charge",
            "FCCAnalyses::ZTauTau::get_jet_total_charge({}, 3)".format(jetClusteringHelper.constituents)
        )

        df = df.Define(
            "jet5_total_charge",
            "FCCAnalyses::ZTauTau::get_jet_total_charge({}, 4)".format(jetClusteringHelper.constituents)
        )

        return df


    # Return the output branches of the analysis
    def output(self):

        # Branches to be exported in the output TTree
        exportBranches = [

            # Lepton variables
            "nlep", "lep_p", "lep_theta", "lep_phi",

            "electrons_isolation", "muons_isolation",

            # Missing energy
            "missing_p", "missing_p_theta", "missing_p_phi",

            # Jet kinematics
            "njets", "jet1_p", "jet2_p",
            "jet1_theta", "jet2_theta",
            "jet1_phi", "jet2_phi",
            "jets_delta_phi", "jets_delta_theta", "jets_delta_r",

            # Tau jet tags
            "jet1_tau_score", "jet2_tau_score",

            # Invariant mass
            "jets_mass", "jet1_mass", "jet2_mass",
            "jets_total_mass", "nprongs",
            
            "jet1_total_charge", "jet2_total_charge",
        ]

        print("Output branches = " + str(exportBranches))
        return exportBranches
    