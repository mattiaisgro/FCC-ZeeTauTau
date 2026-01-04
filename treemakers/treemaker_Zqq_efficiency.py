
#
# Study the tau flavor tagging efficiency on Z -> qq
#
# Test command:
# fccanalysis run --nevents=10 treemaker_Zqq_efficiency.py

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


# The quark flavor to consider ("light" for ud, "charm" for c, "strange" for s, "bottom" for b)
flavor = "charm"


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


print("Quark Flavor = " + flavor)

# The chosen decay of the Z boson
products = ""
flavorShortened = ""

if flavor == "light":
    products = "ud"
    flavorShortened = "l"
elif flavor == "charm":
    products = "cc"
    flavorShortened = "c"
elif flavor == "strange":
    products = "ss"
    flavorShortened = "s"
elif flavor == "bottom":
    products = "bb"
    flavorShortened = "b"


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


# Analysis class to work on the DataFrame
class Analysis():

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
            "p8_ee_Z" + products +"_ecm91": {
                "fraction": 0.005,
                "chunks": 100,
            },
        }

        # Input directory where to find the samples
        self.input_dir = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/"

        # Optional: output directory, default is local running directory
        username = getpass.getuser()
        self.output_dir = f"/eos/user/{username[0]}/{username}/Ztautau/efficiency/{channel}"

        # Title of the analysis
        self.analysis_name = "FCC-ee Z->" + products + " tagger efficiency analysis"

        # Number of threads to run on
        self.n_threads = 4

        # Whether to run on Condor
        self.run_batch = True

        # Whether to use weighted events
        self.do_weighted = True

        # Whether to read the input files with podio::DataSource
        self.use_data_source = False


    # __________________________________________________________
    # Run the analysis and return the resulting DataFrame
    def analyzers(self, df):

        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Electron0","Electron#0.index")
        df = df.Alias("Particle0","Particle#0.index")
        df = df.Alias("MCParticles", "Particle")

        # Get all the leptons from the collection
        df = df.Define(
            "muons_all",
            "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)",
        )

        df = df.Define(
            "electrons_all",
            "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)",
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
            "muons_all.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_p(muons_all)[0] : (electrons_all.size() > 0 ? FCCAnalyses::ReconstructedParticle::get_p(electrons_all)[0] : -999) "
        )

        df = df.Define(
            "nlep",
            "electrons_all.size() + muons_all.size()"
        )


        # Kinematic variables of jets
        df = df.Define("jets_p", "JetClusteringUtils::get_p({})".format(jetClusteringHelper.jets))
        df = df.Define("jets_theta", "JetClusteringUtils::get_theta({})".format(jetClusteringHelper.jets))
        df = df.Define("jets_phi", "JetClusteringUtils::get_phi({})".format(jetClusteringHelper.jets))

        df = df.Define("jet1_p", "jets_p[0]")
        df = df.Define("jet2_p", "jets_p[1]")

        df = df.Define("jet1_theta", "jets_theta[0]")
        df = df.Define("jet2_theta", "jets_theta[1]")

        df = df.Define("jet1_phi", "jets_phi[0]")
        df = df.Define("jet2_phi", "jets_phi[1]")


        # Filter events with exactly 2 jets
        print("FILTER: Filtering events with exactly 2 jets")
        df = df.Filter("jets_p.size() == 2")


        # Filter back-to-back events
        back2backRTolerance = 0.1

        df = df.Define("jets_delta_phi", "std::abs(jet1_phi - jet2_phi)")
        df = df.Define("jets_delta_theta", "std::abs(jet1_theta - jet2_theta)")

        df = df.Define("jets_delta_r", "std::sqrt(jets_delta_phi * jets_delta_phi + jets_delta_theta * jets_delta_theta)")
        print("FILTER: Filtering back-to-back events with delta R tolerance = {}".format(back2backRTolerance))
        df = df.Filter("std::abs(jets_delta_r - 3.14) < {}".format(back2backRTolerance))


        # Match the jet quark flavours
        df = df.Define(
            "jets_flavor_match",
            "JetTaggingUtils::get_flavour({}, Particle)".format(jetClusteringHelper.jets)
        )


        # Select quarks with respect to flavor

        if flavor != "strange":
            df = df.Define(
                "jets_" + flavor +"_match",
                "JetTaggingUtils::get_" + flavorShortened +"tag(jets_flavor_match, 1.0)"
            )
        else:
            # For strange flavor use custom function (no function in FCCAnalyses library)
            df = df.Define("jets_strange_match", "FCCAnalyses::ZTauTau::get_stag(jets_flavor_match, 1.0)")


        # Build score branch for each flavor
        df = df.Define("jets_" + flavor + "_tau_score", "recojet_isTAU_R5[jets_" + flavor + "_match]")

        return df



    # Return the output branches of the analysis
    def output(self):

        # Branches to be exported in the output TTree
        exportBranches = [

            # Lepton variables
            "nlep", "lep_p",

            # Jet kinematics
            "jet1_p", "jet2_p",
            "jet1_theta", "jet2_theta",
            "jet1_phi", "jet2_phi",
            "jets_delta_phi", "jets_delta_theta", "jets_delta_r",

            # Jet flavours
            "jets_" + flavor + "_tau_score",
        ]

        print("Output branches = " + str(exportBranches))
        return exportBranches
    