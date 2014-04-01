package edu.autovolweb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import weka.clusterers.FilteredClusterer;
import weka.core.Instance;
import weka.core.Instances;

public class EMCluster {
	
	/* Static util methods */
	public static int findMaxCluster(double[] distribs) {
		 List<Double> distribsList = new ArrayList<Double>(distribs.length);
		 for (int j = 0; j < distribs.length; j++) {
			 distribsList.add(distribs[j]);
		 }
		 Double maxVal = Collections.max(distribsList);
		 int maxCluster = distribsList.indexOf(maxVal);
		 
		 return maxCluster;
	}
	
	public static ArrayList<EMCluster> createClusterToLabelMap(Instances labeledData, Instances unlabeledData, FilteredClusterer em) {

		try {
			 // associate clusters with ringer labels
			 int[][] clusterLabelCounts = new int[em.numberOfClusters()]
					 [labeledData.classAttribute().numValues()];
			 
			// for (Instance i : unlabeledData) {
			for (int ind = 0; ind < labeledData.numInstances(); ind++) {
				 Instance labeled = labeledData.instance(ind);
				 Instance unlabeled = unlabeledData.instance(ind);
				 double[] distribs = em.distributionForInstance(unlabeled);
				 int maxCluster = EMCluster.findMaxCluster(distribs);
				 clusterLabelCounts[maxCluster][(int) labeled.classValue()]++;
			 }
			 
			 // list(i) -> majority label for ith cluster
			 ArrayList<EMCluster> clusterToLabelMap = 
					 new ArrayList<EMCluster>(em.numberOfClusters());
			 for (int i = 0; i < em.numberOfClusters(); i++) {
				 List<Integer> countList = new ArrayList<Integer>(labeledData.numClasses());
				 long totalCount = 0;
				 for (int j = 0; j < clusterLabelCounts[i].length; j++) {
					 countList.add(clusterLabelCounts[i][j]);
					 totalCount += clusterLabelCounts[i][j];
				 }
				 
				 Integer maxCount = Collections.max(countList);
				 int maxLabel = countList.indexOf(maxCount);
				 String maxLabelString = labeledData.classAttribute().value(maxLabel);
				 EMCluster cluster = new EMCluster();
				 cluster.setRingerLabel(maxLabelString);
		
				 double probOfLabel = maxCount / totalCount;
				 cluster.setProbOfLabel(probOfLabel);
				 clusterToLabelMap.add(i, cluster);
			 }
			 
			 
			 return clusterToLabelMap;

		 } catch (Exception e) {
			 e.printStackTrace();
		 }
		 
		 return null;
	}
	
	/* data object */
	private String ringerLabel;
	private Double probOfLabel;
	
	public EMCluster() {}
	
	public String getRingerLabel() {
		return ringerLabel;
	}

	public void setRingerLabel(String ringerLabel) {
		this.ringerLabel = ringerLabel;
	}

	public Double getProbOfLabel() {
		return probOfLabel;
	}

	public void setProbOfLabel(Double probOfLabel) {
		this.probOfLabel = probOfLabel;
	}

}
