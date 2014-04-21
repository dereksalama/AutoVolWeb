package edu.autovolweb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;

import com.google.gson.JsonObject;

import weka.clusterers.FilteredClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Servlet implementation class ClusterLocKnnClassifyServlet
 */
@WebServlet("/ClusterLocKnnClassifyServlet")
public class ClusterLocKnnClassifyServlet extends LocKnnClassifyServlet {
	private static final long serialVersionUID = 1L;
	
	private static final int NUM_VECTORS_TO_CLUSTER = 8;
	
	private static final int NUM_CLUSTERS = 20;
	
	private Map<String, FilteredClusterer> clustererMap;
	
	@Override 
	public void init() throws ServletException {
		super.init();
		clustererMap = new ConcurrentHashMap<>();
	}

	@Override
	protected Instances loadData(String userId) throws Exception {
		Instances allDataLoc = super.loadData(userId);
		
		Remove r = new Remove();
		r.setAttributeIndices("" + (allDataLoc.classIndex() + 1));
		r.setInputFormat(allDataLoc);
		Instances unlabeledData = Filter.useFilter(allDataLoc, r);
		
		
		SimpleKMeans km = new SimpleKMeans();
		km.setNumClusters(NUM_CLUSTERS);
		km.setPreserveInstancesOrder(true);
		km.setInitializeUsingKMeansPlusPlusMethod(true);
		km.setMaxIterations(500);
		
		FilteredClusterer clusterer = new FilteredClusterer();
		clusterer.setClusterer(km);
		
		Normalize n = new Normalize();
		n.setInputFormat(unlabeledData);
		clusterer.setFilter(n);
		
		clusterer.buildClusterer(unlabeledData);
		clustererMap.put(userId, clusterer);
		
		Instances data = createClusterDataset(unlabeledData.numInstances());
		
		int[] assignments = ((SimpleKMeans) clusterer.getClusterer()).getAssignments();
		for (int i = 0; i < unlabeledData.numInstances() - NUM_VECTORS_TO_CLUSTER; i++) {
			if (allDataLoc.instance(i).value(allDataLoc.attribute("day")) != 
					allDataLoc.instance(i + NUM_VECTORS_TO_CLUSTER - 1).value(allDataLoc.attribute("day"))) {
				continue; // skip until all from same day
			}
			Instance inst = new DenseInstance(NUM_VECTORS_TO_CLUSTER + 1);
			inst.setDataset(data);
			int[] classCounts = new int[data.classAttribute().numValues()];
			for (int j = 0; j < NUM_VECTORS_TO_CLUSTER; j++) {
				classCounts[(int) Math.round(allDataLoc.instance(i + j).classValue())]++;
				int cluster = assignments[i + j];
				inst.setValue(j, cluster);
			}
	    	String label = findMaxLabel(data, classCounts);
			inst.setValue(data.classAttribute(), label);
			
			data.add(inst);
		}
		
		return data;
	}

	private Instances createClusterDataset(int numInstances) {
		ArrayList<Attribute> attrList = new ArrayList<>();
		for (int i = 0; i < NUM_VECTORS_TO_CLUSTER; i++) {
			attrList.add(new Attribute("" + i));
		}
		ArrayList<String> ringerValues = new ArrayList<String>();
		ringerValues.add("silent");
		ringerValues.add("vibrate");
		ringerValues.add("normal");
		Attribute ringerAttr = new Attribute("ringer", ringerValues);
		attrList.add(ringerAttr);
		
		Instances data = new Instances("clustered", attrList, numInstances);
		data.setClass(data.attribute("ringer"));
		return data;
	}

	private String findMaxLabel(Instances data, int[] classCounts) {
		int maxIndex = 0;
		int maxCount = Integer.MIN_VALUE;
		for (int j = 0; j < data.classAttribute().numValues(); j++) {
			if (classCounts[j] > maxCount) {
				maxCount = classCounts[j];
				maxIndex = j;
			}
		}
		String label = data.classAttribute().value(maxIndex);
		return label;
	}

	@Override
	protected Instance constructTarget(String input, String userId) {
		List<CurrentStateData> states = CurrentStateUtil.fromJson(input);
		if (states.size() < NUM_VECTORS_TO_CLUSTER) {
			return null;
		}
		
		List<Instance> instances = new ArrayList<>(NUM_VECTORS_TO_CLUSTER);
		Map<String, Integer> classCounts = new HashMap<>();
		for (CurrentStateData state : states) {
			Instance locTarget = CurrentStateUtil.extractLocInstance(state);
			String locCluster;
			try {
				locCluster = "" + ((int) getLocClusterer(userId).clusterInstance(locTarget));
			} catch (Exception e) {
				e.printStackTrace();
				return null;
			}
			if (!getLocClusters(userId).contains(locCluster)) {
				locCluster = "other";
			}
			
			Instance target = CurrentStateUtil.toUnlabeledLocInstance(state, locCluster, 
					getLocClusters(userId));
			instances.add(target);
			
			Integer count = classCounts.get(state.getRinger());
			Integer newCount = (count == null ? 1 : count + 1);
			classCounts.put(state.getRinger(), newCount);
		}
		
		int maxCount = -1;
		String maxLabel = null;
		for (Entry<String, Integer> e : classCounts.entrySet()) {
			if (e.getValue() > maxCount) {
				maxCount = e.getValue();
				maxLabel = e.getKey();
			}
		}
		Collections.sort(instances, new ViewDataServlet.TimeComparator());
		Instances clusteredDataset = createClusterDataset(0);
		
		try {
			int[] assignments = new int[NUM_VECTORS_TO_CLUSTER];
			for (int i = 0; i < NUM_VECTORS_TO_CLUSTER; i++) {
				Instance inst = instances.get(i);
				int cluster = clustererMap.get(userId).clusterInstance(inst);
				assignments[i] = cluster;
			}
			
			Instance target = new DenseInstance(clusteredDataset.numAttributes());
			target.setDataset(clusteredDataset);
			for (int i = 0; i < NUM_VECTORS_TO_CLUSTER; i++) {
				target.setValue(i, assignments[i]);
			}
			target.setValue(clusteredDataset.attribute("ringer"), maxLabel);
			return target;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
	@Override
	protected String prepareOutput(JsonObject json, Instance target) {
		return json.toString();
	}

}
