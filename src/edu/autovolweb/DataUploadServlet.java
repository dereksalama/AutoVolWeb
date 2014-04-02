package edu.autovolweb;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
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
	
	public static final String CLUSTER_LABELS_FILE = "cluster_labels";
	public static final String EM_MODEL_FILE = "em_model";
    
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
	 
	 private class RetrainRunnable implements Runnable {
		 private final String newData;
		 public RetrainRunnable(String newData) {
			this.newData = newData;
		}
		 
		 @Override
		 public void run() {
			 // interpret incoming data
			 Instances allData = CurrentStateUtil.convertCurrentStateData(newData);

			 // write to today's file
			 DateTime today = new DateTime();
			 String newFileName = "data_" + today.getDayOfMonth() + "_" +
					 today.getMonthOfYear() + "_" + today.getYear();

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
				 String fileName = "data_" + day.getDayOfMonth() + "_" +
						 day.getMonthOfYear() + "_" + day.getYear();
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

			 // EM cluster
			 EM unfilteredEM = new EM();
			 unfilteredEM.setMaximumNumberOfClusters(20);
			 Normalize normalizer = new Normalize();
			 
			 Remove removeClass = new Remove();
			 removeClass.setAttributeIndices("" + (allData.classIndex() + 1));


			 try {
				 removeClass.setInputFormat(allData);
				 Instances dataClassRemoved = Filter.useFilter(allData, removeClass);

				 normalizer.setInputFormat(dataClassRemoved);
				 FilteredClusterer em = new FilteredClusterer();
				 em.setClusterer(unfilteredEM);
				 em.setFilter(normalizer);
				 em.buildClusterer(dataClassRemoved);
				 List<EMCluster> clusterToLabels = EMCluster
						 .createClusterToLabelMap(allData, dataClassRemoved, em);
				 
				 
				 // save cluster labels
				 File clusterToLabelsFile = new File(CLUSTER_LABELS_FILE);
				 BufferedWriter writer = new BufferedWriter(
						 new FileWriter(clusterToLabelsFile, false)); // overwrite
				 
				 Gson gson = new Gson();
				 String json = gson.toJson(clusterToLabels);
				 writer.write(json);
				 writer.close();
				 
				 // save EM model
				 SerializationHelper.write(new FileOutputStream(EM_MODEL_FILE, false), em);
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
		new RetrainRunnable(incomingDataString).run();

		response.setStatus(HttpServletResponse.SC_ACCEPTED);
	}

}
