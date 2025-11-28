import numpy as np
from pymatgen.analysis.local_env import VoronoiNN

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import matplotlib.pyplot as plt
import re
from scipy.stats import kurtosis
from collections import defaultdict

# Use SpacegroupAnalyzer to get symmetry information
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, PointGroupAnalyzer
from pymatgen.core.structure import Molecule
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


class Create_Graphlets:
    """
    A class to create graphlets (local structural features) from a CIF structure.

    The graphlets include 1-site, 2-site, and 3-site features.
    """
     
    def __init__(self, cif_structure,atomic_radii=None):
        prim_structure=cif_structure.get_primitive_structure() # so we only worry about the smallest unit cell
        self.structure = prim_structure
        if(atomic_radii is None):
            from mendeleev import element
            self.atomic_radii = {el.symbol: el.atomic_radius for el in [element(i) for i in range(1, 119)]}
        else:
            self.atomic_radii=atomic_radii
        self.neighb_data=self.get_neighbors()

    
    def remove_non_alphabets(self,input_string):
        return re.sub(r'[^a-zA-Z]', '', input_string)
    
    def get_string(self,composition):
        sorted_elems=sorted(composition.keys())
        species_string=''
        for elem in sorted_elems:
            string=elem+str(np.round(composition[elem],4))
            species_string += string
        return species_string #example:'Ag0.9333Hg0.0667' 
    
    def Max_bond_length(self,composition_1, composition_2,factor=1.5):
        rad_1=sum([composition_1[e]*self.atomic_radii[self.remove_non_alphabets(e)] for e in composition_1])

        rad_2=sum([composition_2[e]*self.atomic_radii[self.remove_non_alphabets(e)] for e in composition_2])
            
        return (rad_1 + rad_2)*factor/100  # Convert pm to Angstrom
        
    def get_neighbors(self):
        """
        Find the nearest neighbors for each site in the structure using Voronoi tessellation.

        Uses VoronoiNN from pymatgen to find the nearest neighbors.

        Returns:
            dict: A dictionary where each key is a site index and the value is a dictionary containing:
                - 'site_elem': Element symbol of the site.
                - 'neighb_sites': List of neighboring site indexes.
                - 'neighb_coords': List of coordinates of neighboring sites.
                - 'neighb_vectors': List of vectors from the site to its neighbors.
                - 'neighb_dists': List of distances to neighboring sites.
                - 'neighb_elems': List of element symbols of neighboring sites.
        """
        voronoi_nn = VoronoiNN()

        
        
        # Find nearest neighbors and store them
        neighb_data=defaultdict(list)
        for i, site in enumerate(self.structure.sites):
            neighbors = voronoi_nn.get_nn_info(self.structure, i)
            site_composition=site.species.as_dict()
            site_label=self.get_string(site_composition)
            neighb_sites = []
            neighb_coords = []
            neighb_vectors = []
            neighb_dists=[]
            neighb_labels=[]
            neighb_compositions=[]

            for neighbor in neighbors:
                vector=neighbor['site'].frac_coords-site.frac_coords
                # Adjust vector for periodic boundary conditions
                n_vect = self.structure.lattice.get_cartesian_coords(vector) 

                # Calculate the distance between the atoms
                n_coord = site.coords + n_vect
                n_dist = np.linalg.norm(n_vect)
                n_composition=neighbor['site'].species.as_dict()
                n_label=self.get_string(n_composition)

                if(n_dist<self.Max_bond_length(site_composition,n_composition)):
                    neighb_sites.append(self.structure.sites[neighbor['site_index']])
                    neighb_coords.append(n_coord)
                    neighb_vectors.append(n_vect)
                    neighb_dists.append(n_dist)
                    neighb_compositions.append(n_composition)
                    neighb_labels.append(n_label)
            

            neighb_data[i]={
                'site_composition':site_composition,
                'site_label':site_label,
                'neighb_sites': neighb_sites,
                'neighb_coords':neighb_coords,
                'neighb_vectors':neighb_vectors,
                'neighb_dists':neighb_dists,
                'neighb_compositions':neighb_compositions,
                'neighb_labels':neighb_labels
                }
        return neighb_data


    def Get_1_site_graphlets(self):
        """
        Generate 1-site graphlets.

        Updates:
            self.one_site_graphlets: A list of dictionaries, each representing a unique 1-site graphlet
                                     with keys:
                                     - 'g_order': Graphlet order (=1 for 1 site)
                                     - 'atom': Element composition of the site
                                     - 'site_sym': Point group symmetry symbol for site
                                     - 'count': Number of times this graphlet occurs in the structure
        """
        all_sites = defaultdict(dict)
        seen_sites = set()
    
        # Find nearest neighbors and determine local symmetry of each site (there is a better way than this, using wyckoff position)

        for i, site in enumerate(self.structure.sites):
            site_composition=site.species.as_dict()
            site_label=self.get_string(site_composition)
            
            if site_label not in seen_sites:
                seen_sites.add(site_label)
                site_dict = {
                        'g_order':1,  #order of graphlet
                        'atom': site_composition,
                        'count': 1
                    }
                all_sites[site_label]=site_dict
            else:
                all_sites[site_label]['count']+=1   
    
        self.one_site_graphlets=list(all_sites.values())
        return None

    def Get_2_site_graphlets(self):
        """
        Generate 2-site graphlets: pairs of atoms connected via bonds.

        For each bond between atoms, store the pair of elements and the bond length.
        Bonds are identified based on neighbor data.

        Updates:
            self.two_site_graphlets: A list of dictionaries, each representing a unique 2-site graphlet
                                     with keys:
                                     - 'g_order': Graphlet order (=2 for 2 site)
                                     - 'atom': Element composition of the central atom
                                     - 'neighbor': Element composition of the neighbor atom
                                     - 'distance': Bond length
                                     - 'count': Number of times this graphlet occurs in the structure
        """
        all_pairs = defaultdict(dict)
        seen_pairs = set()
    
        
        for i, site in enumerate(self.structure.sites):
            neighb_data=self.neighb_data[i]


            site_label=neighb_data['site_label']
            site_composition=neighb_data['site_composition']

            neighb_labels=neighb_data['neighb_labels']
            neighb_compositions=neighb_data['neighb_compositions']
            neighb_dists=neighb_data['neighb_dists']

            for n_label,n_dist,n_composition in zip(neighb_labels,neighb_dists,neighb_compositions):
                pair_tuple = (tuple(sorted([site_label, n_label])), round(n_dist, 1))
                if pair_tuple not in seen_pairs:
                    seen_pairs.add(pair_tuple)
                    pair = {
                        'g_order':2, # order of graphlet
                        'atom': site_composition,
                        'neighbor': n_composition,
                        'distance': n_dist,
                        'count': 1
                    }
                    all_pairs[pair_tuple]=pair
                else:
                    all_pairs[pair_tuple]['count']+=1

        self.two_site_graphlets=list(all_pairs.values())
        return None

        
    


    def Get_3_site_graphlets(self):
        """
        Generate 3-site graphlets by analyzing triplets of atoms (central atom and two neighbors).

        For each pair of neighboring atoms connected to the central atom, calculate the angle between them.
        Stores unique triplets based on elements, bond lengths, and angle.

        Updates:
            self.three_site_graphlets: A list of dictionaries, each representing a unique 3-site graphlet
                                       with keys:
                                       - 'g_order': Graphlet order (=3 for 3 site)
                                       - 'atom': Element composition of the central atom
                                       - 'pair_1': Element composition of first neighbor
                                       - 'distance_1': Bond length to first neighbor
                                       - 'pair_2': Element composition of second neighbor
                                       - 'distance_2': Bond length to second neighbor
                                       - 'angle': Angle between the two bonds
                                       - 'count': Number of times this graphlet occurs in the structure
        """
        def calculate_angle(vector1, vector2):
            # Calculate the dot product and the magnitudes of the vectors
            dot_prod = np.dot(vector1, vector2)
            cos_angle = dot_prod / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle)
            return angle_degrees
    
        
    
        all_triplets = defaultdict(dict)
        seen_triplets = set()

        for i, site in enumerate(self.structure.sites):
            neighb_data=self.neighb_data[i]
            
            atom_i=neighb_data['site_composition']
            label_i=neighb_data['site_label']

            neighb_dists=neighb_data['neighb_dists']
            neighb_compositions=neighb_data['neighb_compositions']
            neighb_labels=neighb_data['neighb_labels']
            neighb_vects=neighb_data['neighb_vectors']
            N_neighbs=len(neighb_labels)
        
            # Loop through all pairs of neighbors to create triplets
            for j in range(N_neighbs):
                dist_j=neighb_dists[j]
                vect_j=neighb_vects[j]
                atom_j=neighb_compositions[j]
                label_j=neighb_labels[j]
                for k in range(j + 1, N_neighbs):
                    dist_k=neighb_dists[k]
                    vect_k=neighb_vects[k]
                    atom_k=neighb_compositions[k]
                    label_k=neighb_labels[k]
                    # Calculate the angle between the vectors
                    angle_jk = calculate_angle(vect_j, vect_k)
                    
            
                    sorted_pairs = sorted([(label_j, dist_j), (label_k, dist_k)], key=lambda x: x[0])
                    sorted_pair_labels = (sorted_pairs[0][0], sorted_pairs[1][0])
                    sorted_pair_dists = (round(sorted_pairs[0][1],1), round(sorted_pairs[1][1],1))
            
                    triplet_tuple = (label_i, sorted_pair_labels,sorted_pair_dists , round(angle_jk/10)*10)
                    if triplet_tuple not in seen_triplets:
                        seen_triplets.add(triplet_tuple)
                        triplet = {
                            'g_order':3,
                            'atom': atom_i, #composition of central atom
                            'pair_1': atom_j, # composition of first neighbor
                            'distance_1': dist_j, 
                            'pair_2': atom_k, # composition of second neighbor
                            'distance_2': dist_k,
                            'angle': angle_jk,
                            'count': 1
                        }
                        all_triplets[triplet_tuple]=triplet
                    else:
                        all_triplets[triplet_tuple]['count']+=1
        self.three_site_graphlets=list(all_triplets.values())
        return None           









    
    def get_features(self,atomic_features_dict):
        """
        Generate feature vectors for the graphlets using atomic properties.

        For each graphlet (1-site, 2-site, 3-site), computes features such as mean, standard deviation, and kurtosis
        of bond length, electroneg etc. within each graphlet, from the provided atomic_features_dict.

        Args:
            atomic_features_dict (dict): A dictionary mapping element symbols to their properties.
                                         For example: {'Fe': {'atomic_number': 26, 'atomic_radius': 126}, ...}

        Updates:
            self.one_site_features: Dictionary of features for 1-site graphlets.
            self.two_site_features: Dictionary of features for 2-site graphlets.
            self.three_site_features: Dictionary of features for 3-site graphlets.
        """
        feat_names=list(list(atomic_features_dict.values())[0].keys())     

        def get_atomic_features(composition): #find average atomic features of the composition
            composition_feature={}
            for feat in feat_names:
                weighted_feature=sum([float(atomic_features_dict[self.remove_non_alphabets(elem)][feat])*float(occup) 
                                      for elem,occup in composition.items()])
                composition_feature[feat]=weighted_feature 
            return composition_feature                   

        
        try:
            if(self.one_site_graphlets):
                one_G_feats_dict=defaultdict(list)
                for subG in self.one_site_graphlets:
                    atom=subG['atom']
                    count=subG['count']


                    # atomic features
                    atom_feats=get_atomic_features(atom) 
                    for feat in feat_names:
                        one_G_feats_dict[feat+'_1_ord']+=[atom_feats[feat]]*count

                self.one_site_features=one_G_feats_dict
        except:
            pass
        
        try:
            if(self.two_site_graphlets):
                two_G_feat_dict=defaultdict(list)
                
                for subG in self.two_site_graphlets:
                    atom_1=subG['atom']
                    atom_1_feats=get_atomic_features(atom_1)
                    
                    atom_2=subG['neighbor']
                    atom_2_feats=get_atomic_features(atom_2)
                    
                    bond_len=subG['distance']
                    count=subG['count']

                    # atomic features
                    for feat in feat_names:
                        feat_mean=feat+'_mean_2_ord'
                        two_G_feat_dict[feat_mean]+=[np.mean([atom_1_feats[feat],atom_2_feats[feat]])]*count

                        feat_std=feat+'_std_2_ord'
                        two_G_feat_dict[feat_std]+=[np.std([atom_1_feats[feat],atom_2_feats[feat]])]*count

                    two_G_feat_dict['bond_len_2_ord']+=[bond_len]*count

                self.two_site_features=two_G_feat_dict
        except:
            pass


        try:
            if(self.three_site_graphlets):
                three_G_feat_dict=defaultdict(list)
        
                for subG in self.three_site_graphlets:
                    atom=subG['atom']
                    atom_feats=get_atomic_features(atom)

                    atom_1=subG['pair_1']
                    atom_1_feats=get_atomic_features(atom_1)
                    bond_len_1=subG['distance_1']

                    atom_2=subG['pair_2']
                    atom_2_feats=get_atomic_features(atom_2)
                    bond_len_2=subG['distance_2']

                    angle=subG['angle']

                    count=subG['count']


                    # atomic features
                    for feat in feat_names:

                        e0=atom_feats[feat]
                        e1=atom_1_feats[feat]
                        e2=atom_2_feats[feat]
                        
                        feat_mean=feat+'_mean_3_ord'
                        three_G_feat_dict[feat_mean]+=[np.mean([e0,e1,e2])]*count

                        feat_std=feat+'_std_3_ord'
                        three_G_feat_dict[feat_std]+=[np.std([e0,e1,e2])]*count

                        feat_kurt=feat+'_kurt_3_ord'
                        if (np.std([e0,e1,e2])>1e-6):
                            three_G_feat_dict[feat_kurt]+=[kurtosis([e0,e1,e2])]*count
                        else:
                            three_G_feat_dict[feat_kurt]+=[0.0]*count
                    # two site features
                    three_G_feat_dict['bond_len_mean_3_ord']+=[np.mean([bond_len_1,bond_len_2])]*count
                    three_G_feat_dict['bond_len_std_3_ord']+=[np.std([bond_len_1,bond_len_2])]*count
                    
                    # three site features
                    three_G_feat_dict['angle_3_ord']+=[angle]*count
                
                self.three_site_features   = three_G_feat_dict     
        except:
            pass

class Graphlet_Analyzer:
    """
    A class to analyze graphlets from a list of different materials and generate histogram features.

    Args:
        graphlet_list (list): List of Create_Graphlets objects for a list of different materials.
        max_order (int): Maximum order of graphlets to consider (1, 2, or 3).
        bin_width_factor (float): Factor to adjust the bin width in histograms. The bin widths are estimated with FD rule
        hist_density (bool): If True, histograms will be density plots (normalized).
    """
    def __init__(self, graphlet_list,max_order=3,bin_width_factor=1.0,hist_density=False):
        self.graphlet_list=graphlet_list
        self.max_order=max_order
        self.bin_width_factor=bin_width_factor
        self.hist_density=hist_density
    
    def get_features_dict(self,graphlet):

        """
        Get the combined features dictionary from a graphlet of a particular material.

        Args:
            graphlet (Create_Graphlets): A graphlet object of a particular material.

        Returns:
            dict: A dictionary containing features from 1-site, 2-site, and/or 3-site graphlets.
        """
        if(self.max_order==3):
            features_dict={**graphlet.one_site_features,
                           **graphlet.two_site_features,
                           **graphlet.three_site_features
                           }
        elif(self.max_order==2):
            features_dict={**graphlet.one_site_features,
                           **graphlet.two_site_features
                           }
                                                                                 
        else:
            features_dict=graphlet.one_site_features
        return features_dict
            
    
    
    def get_bins(self):

        def calculate_bin_widths(data):
            """
            Calculate the ideal bin widths for a histogram using Freedman-Diaconis rules.
            """
            n = len(data)
            if len(np.unique(data)) == 1:
                return 0.1  # if only one data exist, bin width doesnt matter
    
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            # Freedman-Diaconis Rule
            if iqr>0:
                fd_bin_width = 2 * iqr / np.cbrt(n)
            else:
                # using Sturges' rule as a backup
                sturges_bin_width = (max(data) - min(data)) / (np.log2(n) + 1)
                fd_bin_width = sturges_bin_width if sturges_bin_width > 0 else 0.1   
            return fd_bin_width*self.bin_width_factor
    
        all_features=defaultdict(list)
        for graphlet in self.graphlet_list:

            features_dict=self.get_features_dict(graphlet)

            for feature,values in features_dict.items():
                all_features[feature]+=values
            
            
        bin_ranges={}
        bins={}
        for feature, values in all_features.items():
            # feature: site_sym_1_ord values: ['Cs', 'Cs', 'Cs', 'Cs', 'C2v', 'C2v', 'C2v',
            min_val=np.min(values) 
            max_val=np.max(values)
            bin_width=calculate_bin_widths(values)
            bin_ranges[feature]=(min_val,max_val,bin_width)
            bins[feature]=np.arange(min_val-bin_width/2,max_val+3*bin_width/2,bin_width)

        return bins
    
    def get_histogram_features(self,num_bins:int =None):
        """
        Generate histogram features for each graphlet in the list.

        For each feature in the graphlets, computes histograms across all samples.
        Also computes mean and standard deviation (magpie features) for each feature.

        Returns:
       
            tuple: (hist_names, hist_array, feat_name_list, feat_value_list, feat_magpie_name_list, feat_magpie_value_list)
                - hist_names (list): Names of histograms.
                - hist_array (numpy.ndarray): Array of histogram data for all samples. 
                    - Each histogram may have a different actual number of bins; padding with -1
                    is used to ensure consistent dimensions.

                    Shape= (n_samples, n_hists, max_nbins, 2)
                    The last dimension of size 2 will store:
                        - bin midpoints at index 0
                        - bin heights at index 1
                Next are the features where each bin of each histogram is a separate feature
                - feat_name_list (list): List of feature names (histogram bins and site symmetries) for all samples.
                - feat_value_list (list): List of feature values corresponding to feat_name_list.
                
                Next are the features where each histogram is reduced to mean and std (like magpie features)
                
                - feat_magpie_name_list (list): List of magpie feature names (mean and std) for all samples.
                - feat_magpie_value_list (list): List of magpie feature values corresponding to feat_magpie_name_list.
        """

        bins=self.get_bins()
       
        hist_features_dict_list=[] # dict of features, storing the histogram bins, for all materials list
        magpie_features_dict_list=[] # dict of features, with mean and std of the histogram, for all materials list
        
        nbins=[] # to find maximum length of bins of all histograms of all materials
        for graphlet in self.graphlet_list:
            hist_features={}
            magpie_features={}
            
            features_dict=self.get_features_dict(graphlet)
 
            for feature,values in features_dict.items():
                if(num_bins):
                    range=(bins[feature][0],bins[feature][-1])
                    bin_heights,bin_edges=np.histogram(values,bins=num_bins,range=range,density=self.hist_density)
                else:
                    bin_heights,bin_edges=np.histogram(values,bins=bins[feature],density=self.hist_density)
                        
                bin_mids=(bin_edges[0:-1]+bin_edges[1:])*0.5
                hist_features[feature]=(bin_mids,bin_heights)
                nbins.append(len(bin_mids))
                val_mean,val_std=np.mean(values),np.std(values)
                cumulants=[1,2]
                magpie_features[feature]=(cumulants, [val_mean,val_std])
                
            hist_features_dict_list.append(hist_features)
            magpie_features_dict_list.append(magpie_features)
            
        self.hist_features_dict_list=hist_features_dict_list
        self.magpie_features_dict_list=magpie_features_dict_list
        self.max_nbins=max(nbins) 


        feat_bin_name_list=[] # list of bin feature names, for all materials list
        feat_bin_value_list=[] #list of bin feature values, for all materials list
        

        for hist_dict in hist_features_dict_list:
            feat_bin_name=[]
            feat_bin_value=[]
            for feat, (bin_mid,bin_height) in hist_dict.items():
                feat_bin_name+=[feat+'='+str(i) for i in bin_mid]
                feat_bin_value+=list(bin_height)
            
            feat_bin_name_list.append(feat_bin_name)
            feat_bin_value_list.append(feat_bin_value)
        # ensure all feature_bin_names are stored in same order for all materials
        feat_bin_names=feat_bin_name_list[0]
        sorted_feat_bin_value_list = []  # Store the sorted values
        for names, vals in zip(feat_bin_name_list,feat_bin_value_list):
            if names!=feat_bin_names:
                indices = [names.index(item) for item in feat_bin_names]
                sorted_vals=[vals[i] for i in indices]
                sorted_feat_bin_value_list.append(sorted_vals)
            else:
                sorted_feat_bin_value_list.append(vals)
        feat_bin_name_list=[feat_bin_names]*len(feat_bin_name_list)
        feat_bin_value_list=sorted_feat_bin_value_list

        # create magpie like features (each histogram simplified to mean and std)
        feat_magpie_name_list=[] #list of mean and std names, for all materials list
        feat_magpie_value_list=[] #list of mean and std values, for all materials list
        for magpie_dict in magpie_features_dict_list:
            feat_name=[]
            feat_value=[]
            for feat, (cumulant,mean_std) in magpie_dict.items():
                feat_name+=[feat+'_cumulant='+str(i) for i in cumulant]
                feat_value+=list(mean_std)
            
            feat_magpie_name_list.append(feat_name)
            feat_magpie_value_list.append(feat_value)
        # ensure all feature_magpie_names are stored in same order for all materials
        feat_magpie_names=feat_magpie_name_list[0]
        sorted_feat_magpie_value_list = []  # Store the sorted values
        for names, vals in zip(feat_magpie_name_list,feat_magpie_value_list):
            if names!=feat_magpie_names:
                indices = [names.index(item) for item in feat_magpie_names]
                sorted_vals=[vals[i] for i in indices]
                sorted_feat_magpie_value_list.append(sorted_vals)
            else:
                sorted_feat_magpie_value_list.append(vals)
        feat_magpie_name_list=[feat_magpie_names]*len(feat_magpie_name_list)
        feat_magpie_value_list=sorted_feat_magpie_value_list



                
        

        # Array to store histogram data (bin midpoints and heights)
        # Shape of the array: (n_samples, n_hists, max_nbins, 2)
        # - n_samples: Number of samples
        # - n_hists: Number of histogram features
        # - max_nbins: Maximum number of bins in any histogram
        #   The extra bins and values to meet the shape, are padded with -1
        # The last dimension of size 2 will store:
        # - bin midpoints at index 0
        # - bin heights at index 1

        self.n_samples=len(hist_features_dict_list)
        hist_names=list(hist_features_dict_list[0].keys()) # names of of each histogram plots
        self.n_hists=len(hist_names)
        hist_array=np.full((self.n_samples,self.n_hists,self.max_nbins,2),-1.0,dtype=float)

        for ns,hist_dict in enumerate(hist_features_dict_list):
            for h_name, (bin_mid,bin_height) in hist_dict.items():
                nh=hist_names.index(h_name)
                nbins=len(bin_mid)
                
                hist_array[ns,nh,0:nbins,0]=bin_mid
                hist_array[ns,nh,0:nbins,1]=bin_height


        return hist_names,hist_array, feat_bin_name_list,feat_bin_value_list,feat_magpie_name_list,feat_magpie_value_list
    
    
    
    
        