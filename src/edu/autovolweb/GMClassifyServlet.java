package edu.autovolweb;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

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
	
	private FilteredClusterer em;
	List<EMCluster> clusterToLabels;
	
	private SimpleKMeans locClusterer;
	private List<String> locClusters;

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String input = request.getParameter("target");
		Gson gson = new Gson();
		CurrentStateData state = gson.fromJson(input, CurrentStateData.class);

		
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
	
    @Override
    public void init() throws ServletException {
    	try {
    		// Load clusterer
			em = (FilteredClusterer) SerializationHelper.read(new FileInputStream(
					DataUploadServlet.EM_MODEL_FILE));
			
			// Load cluster label mapping
			String json = getFileJson(DataUploadServlet.CLUSTER_LABELS_FILE);
			
			Gson gson = new Gson();
			Type collectionType = new TypeToken<List<EMCluster>>(){}.getType();
			clusterToLabels = gson.fromJson(json, collectionType);
			
			locClusterer = (SimpleKMeans) SerializationHelper.read(new FileInputStream(
					DataUploadServlet.LOC_CLUSTERER_FILE));

			String locJson = getFileJson(DataUploadServlet.LOC_DATA_FILE);
			Type locCollectionType = new TypeToken<List<String>>(){}.getType();
			locClusters = gson.fromJson(locJson, locCollectionType);
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			throw new ServletException();
		} catch (Exception e) {
			e.printStackTrace();
			throw new ServletException();
		}
    }

	private String getFileJson(String filename) throws IOException {
		byte[] encoded = Files.readAllBytes(Paths.get(filename));
		String json = Charset.defaultCharset().decode(
				ByteBuffer.wrap(encoded)).toString();
		return json;
	}

}
