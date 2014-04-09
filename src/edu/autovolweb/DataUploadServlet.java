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
import java.util.HashSet;
import java.util.List;
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
	
	// TODO
	public static final int NUM_LOC_CLUSTERS = 1;
	
	public static final String CLUSTER_LABELS_FILE = "cluster_labels";
	public static final String EM_MODEL_FILE = "em_model";
	public static final String LOC_DATA_FILE = "loc_data";
	public static final String LOC_CLUSTERER_FILE = "loc_model";
    
	private ExecutorService executor;
	
	 @Override
	 public void init() throws ServletException {
	    	executor = Executors.newCachedThreadPool();
	 }
	 
	 @Override
	 public void destroy() {
	    	try {
				executor.awaitTermination(5, TimeUnit.MINUTES);
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
		 private final String newData;
		 private final String userId;
		 public RetrainRunnable(String newData, String userId) {
			this.newData = newData;
			this.userId = userId;
		}
		 
		 @Override
		 public void run() {
			 // interpret incoming data
			 Instances allData = CurrentStateUtil.convertCurrentStateData(newData);

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

			 // load previous data (up to how old?)
			 for (int i = 1; i < DATA_AGE; i++) {
				 DateTime day = today.minusDays(i);
				 String fileName = constructArffFileName(day, userId, DataUploadServlet.this);
				 File f = new File(fileName);
				 if (f.exists()) {
					 try {
						 ArffLoader loader = new ArffLoader();
						 loader.setFile(f);
						 Instances moreData = loader.getDataSet();
						 allData.addAll(moreData);
					 } catch (FileNotFoundException e) {
						 e.printStackTrace();
					 } catch (IOException e) {
						 e.printStackTrace();
					 }
				 } else {
					 break; // have gone past oldest file
				 }
			 }



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

				 
				 // EM cluster
				 EM unfilteredEM = new EM();
				 Normalize normalizer = new Normalize();
				 
				 Remove removeClass = new Remove();
				 removeClass.setAttributeIndices("" + (allDataLoc.classIndex() + 1));
				 removeClass.setInputFormat(allDataLoc);
				 Instances dataClassRemoved = Filter.useFilter(allDataLoc, removeClass);

				 normalizer.setInputFormat(dataClassRemoved);
				 FilteredClusterer em = new FilteredClusterer();
				 em.setClusterer(unfilteredEM);
				 em.setFilter(normalizer);
				 em.buildClusterer(dataClassRemoved);
				 List<EMCluster> clusterToLabels = EMCluster
						 .createClusterToLabelMap(allDataLoc, dataClassRemoved, em);
				 
				 
				 // save cluster labels
				 File clusterToLabelsFile = new File(constructUserFileName(userId, CLUSTER_LABELS_FILE));
				 BufferedWriter writer = new BufferedWriter(
						 new FileWriter(clusterToLabelsFile, false)); // overwrite
				 

				 String json = gson.toJson(clusterToLabels);
				 writer.write(json);
				 writer.close();
				 
				 // save EM model
				 SerializationHelper.write(new FileOutputStream(
						 constructUserFileName(userId, EM_MODEL_FILE), 
						 false), em);
				 
			 } catch (Exception e) {
				 e.printStackTrace();
			 }
		 }
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
		//TODO: put back after debugging
		//executor.submit(new RetrainRunnable(incomingDataString));
		new RetrainRunnable(incomingDataString, userId).run();

		response.setStatus(HttpServletResponse.SC_ACCEPTED);
	}

}
