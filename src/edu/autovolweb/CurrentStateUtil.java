package edu.autovolweb;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import weka.clusterers.FilteredClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

// Use this to convert to Instance so we don't need Weka lib on android
public class CurrentStateUtil {
	
	private static final int LOC_CLUSTERS = 20;

	private static Collection<CurrentStateData> fromJson(String json) {
		Gson gson = new Gson();
		TypeToken<Set<CurrentStateData>> typeToken = 
				new TypeToken<Set<CurrentStateData>>(){};
		Type collectionType = typeToken.getType();
		Set<CurrentStateData> data = gson.fromJson(json, collectionType);
		return data;
	}
	
	private static void addInstance(CurrentStateData state, Instances dataset) {
		Instance i = toInstance(state, dataset);
		dataset.add(i);
	}
	
	public static Instance toInstance(CurrentStateData state) {
		return toInstance(state, createDataset());
	}
	
	public static Instance toUnlabeledInstance(CurrentStateData state) {
		return toUnlabeledInstance(state, createUnlabeledDataset());
	}
	
	private static Instance toInstance(CurrentStateData state, Instances dataset) {
		Instance i = new DenseInstance(CurrentStateData.NUM_ATTRS);
		i.setDataset(dataset);
		
		i.setValue(dataset.attribute("day"), state.getDay());
		i.setValue(dataset.attribute("time"), state.getTime());
		i.setValue(dataset.attribute("lat"), state.getLat());
		i.setValue(dataset.attribute("lon"), state.getLon());
		i.setValue(dataset.attribute("loc_provider"), state.getLocProvider());
		i.setValue(dataset.attribute("light"), state.getLight());
		i.setValue(dataset.attribute("distance"), state.getLat());
		i.setValue(dataset.attribute("wifi_count"), state.getWifiCount());
		i.setValue(dataset.attribute("bt_count"), state.getBtCount());
		i.setValue(dataset.attribute("charging"), state.getCharging());
		i.setValue(dataset.attribute("activity_type"), state.getActivityType());
		i.setValue(dataset.attribute("activity_confidence"), state.getActivityConfidence());
		i.setValue(dataset.attribute("ringer"), state.getRinger());
		
		return i;
	}
	
	private static Instance toUnlabeledInstance(CurrentStateData state, Instances dataset) {
		Instance i = new DenseInstance(CurrentStateData.NUM_ATTRS - 1);
		i.setDataset(dataset);
		
		i.setValue(dataset.attribute("day"), state.getDay());
		i.setValue(dataset.attribute("time"), state.getTime());
		i.setValue(dataset.attribute("lat"), state.getLat());
		i.setValue(dataset.attribute("lon"), state.getLon());
		i.setValue(dataset.attribute("loc_provider"), state.getLocProvider());
		i.setValue(dataset.attribute("light"), state.getLight());
		i.setValue(dataset.attribute("distance"), state.getLat());
		i.setValue(dataset.attribute("wifi_count"), state.getWifiCount());
		i.setValue(dataset.attribute("bt_count"), state.getBtCount());
		i.setValue(dataset.attribute("charging"), state.getCharging());
		i.setValue(dataset.attribute("activity_type"), state.getActivityType());
		i.setValue(dataset.attribute("activity_confidence"), state.getActivityConfidence());
		
		return i;
	}
	
	private static Instances createDataset() {
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		
		attributes.add(new Attribute("day"));
		attributes.add(new Attribute("time"));
		attributes.add(new Attribute("lat"));
		attributes.add(new Attribute("lon"));
		
		ArrayList<String> locProviderValues = new ArrayList<String>();
		locProviderValues.add("gps");
		locProviderValues.add("network");
		locProviderValues.add("fused");
		attributes.add(new Attribute("loc_provider", locProviderValues));
		
		attributes.add(new Attribute("light"));
		attributes.add(new Attribute("distance"));
		attributes.add(new Attribute("wifi_count"));
		attributes.add(new Attribute("bt_count"));
		
		ArrayList<String> chargingValues = new ArrayList<String>();
		chargingValues.add("true");
		chargingValues.add("false");
		attributes.add(new Attribute("charging", chargingValues));
		
		ArrayList<String> activityValues = new ArrayList<String>();
		activityValues.add("activity_vehicle");
		activityValues.add("activity_bike");
		activityValues.add("activity_foot");
		activityValues.add("activity_still");
		activityValues.add("activity_unknown");
		activityValues.add("activity_tilting");
		attributes.add(new Attribute("activity_type", activityValues));
		
		attributes.add(new Attribute("activity_confidence"));
		
		ArrayList<String> ringerValues = new ArrayList<String>();
		ringerValues.add("silent");
		ringerValues.add("vibrate");
		ringerValues.add("normal");
		Attribute ringerAttr = new Attribute("ringer", ringerValues);
		attributes.add(ringerAttr);
		
		Instances data = new Instances("Training", attributes, 0);
		data.setClass(ringerAttr);
		
		return data;
	}
	
	private static Instances createUnlabeledDataset() {
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		
		attributes.add(new Attribute("day"));
		attributes.add(new Attribute("time"));
		attributes.add(new Attribute("lat"));
		attributes.add(new Attribute("lon"));
		
		ArrayList<String> locProviderValues = new ArrayList<String>();
		locProviderValues.add("gps");
		locProviderValues.add("network");
		locProviderValues.add("fused");
		attributes.add(new Attribute("loc_provider", locProviderValues));
		
		attributes.add(new Attribute("light"));
		attributes.add(new Attribute("distance"));
		attributes.add(new Attribute("wifi_count"));
		attributes.add(new Attribute("bt_count"));
		
		ArrayList<String> chargingValues = new ArrayList<String>();
		chargingValues.add("true");
		chargingValues.add("false");
		attributes.add(new Attribute("charging", chargingValues));
		
		ArrayList<String> activityValues = new ArrayList<String>();
		activityValues.add("activity_vehicle");
		activityValues.add("activity_bike");
		activityValues.add("activity_foot");
		activityValues.add("activity_still");
		activityValues.add("activity_unknown");
		activityValues.add("activity_tilting");
		attributes.add(new Attribute("activity_type", activityValues));
		
		attributes.add(new Attribute("activity_confidence"));
		
		Instances data = new Instances("Training", attributes, 0);
		
		return data;
	}
	
	public static Instances convertCurrentStateData(String json) {
		Instances data = createDataset();
		
		Collection<CurrentStateData> states = fromJson(json);
		for (CurrentStateData state : states) {
			addInstance(state, data);
		}
		
		return data;
	}
	
	public static Instances extractLocationData(Instances input, boolean oldFormat) {
		ArrayList<Attribute> attributes = new ArrayList<>();
		attributes.add(new Attribute("lat"));
		attributes.add(new Attribute("lon"));
		
		ArrayList<String> locProviderValues = new ArrayList<String>();
		if (oldFormat) {
			locProviderValues.add("0.0");
			locProviderValues.add("1.0");
		} else {
			locProviderValues.add("gps");
			locProviderValues.add("network");
			locProviderValues.add("fused");
		}
		
		attributes.add(new Attribute("loc_provider", locProviderValues));
		Instances locData = new Instances("LocData", attributes, input.numInstances());
		
		for (Instance orig : input) {
			Instance loc = new DenseInstance(3);
			loc.setDataset(locData);
			
			loc.setValue(locData.attribute("lat"), orig.value(input.attribute("lat")));
			if (oldFormat) {
				loc.setValue(locData.attribute("lon"), orig.value(input.attribute("long")));
			} else {
				loc.setValue(locData.attribute("lon"), orig.value(input.attribute("lon")));
			}
			loc.setValue(locData.attribute("loc_provider"), orig.value(input.attribute("loc_provider")));
			
			locData.add(loc);
		}
		
		return locData;
	}
	
	public static FilteredClusterer trainLocationClusterer(Instances locData) throws Exception {
		SimpleKMeans km = new SimpleKMeans();
		km.setInitializeUsingKMeansPlusPlusMethod(true);
		km.setPreserveInstancesOrder(true);
		km.setNumClusters(LOC_CLUSTERS); // ?
		
		FilteredClusterer clusterer = new FilteredClusterer();
		Normalize normalizer = new Normalize();
		normalizer.setInputFormat(locData);
		clusterer.setFilter(normalizer);
		clusterer.setClusterer(km);
		
		clusterer.buildClusterer(locData);
		
		return clusterer;
	}
	
	public static List<String> findTopClusters(SimpleKMeans km, Instances locData) {
		int[] clusterSizes = km.getClusterSizes();
		
		// Find clusters that make top 90%
		int[] sortedSizes = Arrays.copyOf(clusterSizes, clusterSizes.length);
		Arrays.sort(sortedSizes);
		int totalCount = locData.numInstances();
		int sum = 0;
		int include = 0;
		for (int i = sortedSizes.length - 1; i >= 0; i--) {
			sum += sortedSizes[i];
			if (((double) sum / totalCount) >= .90 ) {
				include = i;
				break;
			}
		}
		
		Set<Integer> locClusters = new HashSet<Integer>(LOC_CLUSTERS - include);

		for (int i = sortedSizes.length - 1; i >= include; i--) {
			int candidateCount = sortedSizes[i];
			int index = find(clusterSizes, candidateCount);
			locClusters.add(index);
		}
		
		int numNewClusters = LOC_CLUSTERS - include;
		List<String> clusterNames = new ArrayList<String>(numNewClusters);
		for (Integer i : locClusters) {
			clusterNames.add(i.toString());
		}
		clusterNames.add("other");

		return clusterNames;
	}
	
	public static Instances replaceLocationData(Instances input, int[] locIndices, 
			List<String> clusterNames, int[] assignments) throws Exception {
		Remove removeLoc = new Remove();
		removeLoc.setAttributeIndicesArray(locIndices);
		removeLoc.setInputFormat(input);
		
		Set<String> lookupSet = new HashSet<String>(clusterNames.size());
		lookupSet.addAll(clusterNames);

		Instances result = Filter.useFilter(input, removeLoc);
		Attribute locAttr = new Attribute("loc", clusterNames);
		result.insertAttributeAt(locAttr, result.numAttributes() - 1);

		for (int i = 0; i < assignments.length; i++) {
			String cluster = "" + assignments[i];
			if (lookupSet.contains(cluster)) {
				result.instance(i).setValue(result.attribute("loc"), cluster);
			} else {
				result.instance(i).setValue(result.attribute("loc"), "other");
			}
		}
		
		return result;
	}
	
	private static int find(int[] array, int value) {
	    for(int i=0; i<array.length; i++) {
	         if(array[i] == value) {
	             return i;
	         }
	    }
	    
	    return -1;
	}
}
