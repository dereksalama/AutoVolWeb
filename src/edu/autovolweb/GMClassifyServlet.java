package edu.autovolweb;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import weka.clusterers.FilteredClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.SerializationHelper;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

/**
 * Servlet implementation class GMClassifyServlet
 */
@WebServlet("/GMClassifyServlet")
public class GMClassifyServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;
	
	private Map<String, FilteredClusterer> emMap;
	private Map<String, List<EMCluster>> clusterToLabelsMap;
	
	private Map<String, SimpleKMeans> locClustererMap;
	private Map<String, List<String>> locClustersMap;
	
	private Set<String> initializedUsers;
	
	@Override
	public void init() throws ServletException {
		emMap = new ConcurrentHashMap<>();
		clusterToLabelsMap = new ConcurrentHashMap<>();
		
		locClustererMap = new ConcurrentHashMap<>();
		locClustersMap = new ConcurrentHashMap<>();
		
		initializedUsers = Collections.synchronizedSet(new HashSet<String>());
	}
	
	private void initForUser(String userId) throws ServletException {
    	try {
    		// Load clusterer
			FilteredClusterer em = (FilteredClusterer) SerializationHelper.read(new FileInputStream(
					DataUploadServlet.constructUserFileName(userId, DataUploadServlet.EM_MODEL_FILE)));
			emMap.put(userId, em);
			
			// Load cluster label mapping
			String json = getFileJson(userId, DataUploadServlet.CLUSTER_LABELS_FILE);
			
			Gson gson = new Gson();
			Type collectionType = new TypeToken<List<EMCluster>>(){}.getType();
			List<EMCluster> clusterToLabels = gson.fromJson(json, collectionType);
			clusterToLabelsMap.put(userId, clusterToLabels);
			
			SimpleKMeans locClusterer = (SimpleKMeans) SerializationHelper.read(new FileInputStream(
					DataUploadServlet.constructUserFileName(userId,DataUploadServlet.LOC_CLUSTERER_FILE)));
			locClustererMap.put(userId, locClusterer);

			String locJson = getFileJson(userId, DataUploadServlet.LOC_DATA_FILE);
			Type locCollectionType = new TypeToken<List<String>>(){}.getType();
			List<String> locClusters = gson.fromJson(locJson, locCollectionType);
			locClustersMap.put(userId, locClusters);
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			throw new ServletException();
		} catch (Exception e) {
			e.printStackTrace();
			throw new ServletException();
		}
	}

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String userId = request.getParameter("user");
		if (userId == null) {
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
		}
		if (!initializedUsers.contains(userId)) {
			initForUser(userId);
			initializedUsers.add(userId);
		}
		
		String input = request.getParameter("target");
		if (input == null) {
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
		}
		Gson gson = new Gson();
		CurrentStateData state = gson.fromJson(input, CurrentStateData.class);

		FilteredClusterer em = emMap.get(userId);
		List<EMCluster> clusterToLabels = clusterToLabelsMap.get(userId);
		SimpleKMeans locClusterer = locClustererMap.get(userId);
		List<String> locClusters = locClustersMap.get(userId);
		
		try {
			
			Instance locTarget = CurrentStateUtil.extractLocInstance(state);
			String locCluster = "" + ((int) locClusterer.clusterInstance(locTarget));
			if (!locClusters.contains(locCluster)) {
				locCluster = "other";
			}
			Instance target = CurrentStateUtil.toUnlabeledLocInstance(state,
					locCluster, locClusters);
					
			double[] distrib = em.distributionForInstance(target);
			int maxCluster = em.clusterInstance(target);
			EMCluster cluster = clusterToLabels.get(maxCluster);
			String label = cluster.getRingerLabel();
			double probOfCluster = distrib[maxCluster];
			double probOfLabel = cluster.getProbOfLabel();
			
			JsonObject json = new JsonObject();
			json.addProperty("label", label);
			json.addProperty("prob_cluster", probOfCluster);
			json.addProperty("prob_label", probOfLabel);
			json.addProperty("loc_cluster", target.value(target.numAttributes() - 1));
			response.getWriter().write(json.toString());
		} catch (Exception e) {
			e.printStackTrace();
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
		}
	}
	

	private String getFileJson(String userId, String filename) throws IOException {
		String fullFilename = DataUploadServlet.constructUserFileName(userId, filename);
		byte[] encoded = Files.readAllBytes(Paths.get(fullFilename));
		String json = Charset.defaultCharset().decode(
				ByteBuffer.wrap(encoded)).toString();
		return json;
	}

}
