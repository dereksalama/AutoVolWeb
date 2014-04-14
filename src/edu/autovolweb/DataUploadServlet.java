package edu.autovolweb;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.joda.time.DateTime;

import weka.clusterers.EM;
import weka.clusterers.FilteredClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import com.google.gson.Gson;

/**
 * Servlet implementation class DataUploadServlet
 */
@WebServlet("/DataUploadServlet")
public class DataUploadServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;
	
	public static final int DATA_AGE = 14; //keep data for two weeks
	
	//TODO
	public static final int NUM_LOC_CLUSTERS = 20;
	
	public static final String CLUSTER_LABELS_FILE = "cluster_labels";
	public static final String EM_MODEL_FILE = "em_model";
	public static final String LOC_DATA_FILE = "loc_data";
	public static final String LOC_CLUSTERER_FILE = "loc_model";
	
	public static final String AVG_CLUSTER_LABELS_FILE = "cluster_labels";
	public static final String AVG_EM_MODEL_FILE = "em_model";
	
	private static final int NUM_VECTORS_TO_AVG = 4;
    
	private ExecutorService executor;
	
	 @Override
	 public void init() throws ServletException {
	    	executor = Executors.newCachedThreadPool();
	 }
	 
	 @Override
	 public void destroy() {
	    	try {
				executor.awaitTermination(1, TimeUnit.MINUTES);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
	 }
	 
	 public static String constructArffFileName(DateTime day, String user, HttpServlet servlet) {
		 String safeNameConv = convertUser(user);
		 String filename = safeNameConv + "_" + day.getDayOfMonth() + "_" +
				 day.getMonthOfYear() + "_" + day.getYear(); 
		 return filename;
	 }
	 
	 private static String convertUser(String user)  {
		 String safeName = user.replaceAll("\\W+", "");
		 String safeNameConv = null;
		 try {
			 safeNameConv = new String(safeName.getBytes("UTF-8"), Charset.defaultCharset());
		 } catch (UnsupportedEncodingException e) {
			 e.printStackTrace();
		 }
		 return safeNameConv;
	 }
	 
	 public static String constructUserFileName(String userId, String filename) {
		 return convertUser(userId) + "_" + filename;
	 }
	 
	 private class RetrainRunnable implements Runnable {
		 private final String userId;
		 public RetrainRunnable(String userId) {
			this.userId = userId;
		}
		 
		 @Override
		 public void run() {

			 Instances allData = ViewDataServlet.loadAllData(userId, DataUploadServlet.this);
			 try {
				 // Loc stuff
				 Instances locData = CurrentStateUtil.extractLocationData(allData, false);
				 SimpleKMeans locClusterer = CurrentStateUtil.trainUnfilteredLocationClusterer(locData, 
						 NUM_LOC_CLUSTERS);
				 List<String> topClusterList = CurrentStateUtil.findTopClusters(locClusterer, 
						 allData.numInstances());
				 Set<String> locClusters = new HashSet<>(topClusterList.size());
				 locClusters.addAll(topClusterList);

				 Instances allDataLoc = CurrentStateUtil.replaceLocationData(allData, 
						 new int[] {2,3,4}, topClusterList, 
						 locClusterer.getAssignments());
				 allDataLoc.setClass(allDataLoc.attribute("ringer"));
				 SerializationHelper.write(new FileOutputStream(
						 constructUserFileName(userId, LOC_CLUSTERER_FILE), false), 
						 locClusterer);

				 Gson gson = new Gson();
				 File locClustersFile = new File(
						 constructUserFileName(userId, LOC_DATA_FILE));
				 BufferedWriter locWriter = new BufferedWriter(
						 new FileWriter(locClustersFile, false));
				 String locJson = gson.toJson(topClusterList);
				 locWriter.write(locJson);
				 locWriter.close();
				 
				 buildStandardModel(allDataLoc, CLUSTER_LABELS_FILE, EM_MODEL_FILE);
				 buildAvgModel(allDataLoc);
				 
			 } catch (Exception e) {
				 e.printStackTrace();
			 }
		 }

		private void buildStandardModel(Instances allDataLoc, String clustersFile, String modelFile)
				throws Exception, IOException, FileNotFoundException {
			 Remove removeClass = new Remove();
			 removeClass.setAttributeIndices("" + (allDataLoc.classIndex() + 1));
			 removeClass.setInputFormat(allDataLoc);
			 Instances dataClassRemoved = Filter.useFilter(allDataLoc, removeClass);
			 
			 
			// EM cluster
			 EM unfilteredEM = new EM();
			 Normalize normalizer = new Normalize();
			 
			 normalizer.setInputFormat(dataClassRemoved);
			 FilteredClusterer em = new FilteredClusterer();
			 em.setClusterer(unfilteredEM);
			 em.setFilter(normalizer);
			 em.buildClusterer(dataClassRemoved);
			 List<EMCluster> clusterToLabels = EMCluster
					 .createClusterToLabelMap(allDataLoc, dataClassRemoved, em);
			 
			 
			 // save cluster labels
			 File clusterToLabelsFile = new File(constructUserFileName(userId, clustersFile));
			 BufferedWriter writer = new BufferedWriter(
					 new FileWriter(clusterToLabelsFile, false)); // overwrite
			 
			 Gson gson = new Gson();
			 String json = gson.toJson(clusterToLabels);
			 writer.write(json);
			 writer.close();
			 
			 // save EM model
			 SerializationHelper.write(new FileOutputStream(
					 constructUserFileName(userId, modelFile), 
					 false), em);
		}
		
		private void buildAvgModel(Instances allDataLoc)
				throws Exception, IOException, FileNotFoundException {
			Instances avgData = new Instances(allDataLoc, allDataLoc.numInstances() / 
					NUM_VECTORS_TO_AVG);
			LinkedList<Instance> queue = new LinkedList<>();
			for (Instance i : allDataLoc) {
				queue.addFirst(i);
				if (queue.size() > NUM_VECTORS_TO_AVG) {
					queue.removeLast();
				} else if (queue.size() == NUM_VECTORS_TO_AVG) {
					Instance avg = avgInstances(queue, avgData);
					avgData.add(avg);
				}
			}
			
			buildStandardModel(avgData, AVG_CLUSTER_LABELS_FILE, AVG_EM_MODEL_FILE);
		}
	 }
	 
	 public static Instance avgInstances(List<Instance> instances, Instances dataset) {
		 Instance result = new DenseInstance(dataset.numAttributes());
		 result.setDataset(dataset);
		 
		 for (int i = 0; i < dataset.numAttributes(); i++) {
			 Attribute attr = dataset.attribute(i);
			 if (attr.isNumeric()) {
				 double sum = 0;
				 for (Instance inst : instances) {
					 sum += inst.value(attr);
				 }
				 
				 result.setValue(attr, sum / instances.size());
			 } else {
				 Map<Double, Integer> countMap = new HashMap<>();
				 for (Instance inst : instances) {
					 Double val = inst.value(attr);
					 Integer count = countMap.get(val);
					 if (count == null) {
						 countMap.put(val, 1);
					 } else {
						 countMap.put(val, count + 1);
					 }
				 }
				 
				 int maxCount = -1;
				 double maxVal = 0;
				 for (Entry<Double, Integer> e : countMap.entrySet()) {
					 if (e.getValue() >= maxCount) {
						 maxCount = e.getValue();
						 maxVal = e.getKey();
					 }
				 }
				 
				 result.setValue(attr, maxVal);
			 }
		 }
		 
		 return result;
	 }
	 
	

	/**
	 * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		
		//final String incomingDataString = request.getParameter("data");
		StringBuffer buffer = new StringBuffer();
		String line = null;

		BufferedReader reader = request.getReader();
		String userId = reader.readLine(); // first line should be user id
		if (userId == null) {
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
			return;
		}
		
		while ((line = reader.readLine()) != null) {
			buffer.append(line);
		}

		String incomingDataString = buffer.toString();
		if (incomingDataString == null) {
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
			return;
		}
		
		 // interpret incoming data
		 Instances allData = CurrentStateUtil.convertCurrentStateData(incomingDataString);
		 allData.sort(allData.attribute("time"));

		 // write to today's file
		 DateTime today = new DateTime();
		 String newFileName = constructArffFileName(today, userId, DataUploadServlet.this);

		 File newFile = new File(newFileName);
		 if (newFile.exists()) {
			 ArffLoader loader = new ArffLoader();
			 try {
				loader.setFile(newFile);
				Instances oldData = loader.getDataSet();
				allData.addAll(oldData);
			} catch (IOException e) {
				e.printStackTrace();
			}
		 }

		 ArffSaver saver = new ArffSaver();
		 saver.setInstances(allData);
		 try {
			saver.setFile(newFile);
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}
		//TODO: put back after debugging
		//executor.submit(new RetrainRunnable(userId));
		//new RetrainRunnable(incomingDataString, userId).run();

		response.setStatus(HttpServletResponse.SC_ACCEPTED);
	}

}
