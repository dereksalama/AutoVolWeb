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
import weka.core.Instance;
import weka.core.SerializationHelper;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

/**
 * Servlet implementation class GMClassifyServlet
 */
@WebServlet("/GMClassifyServlet")
public class GMClassifyServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;
	
	private FilteredClusterer em;
	List<EMCluster> clusterToLabels;

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String json = request.getParameter("target");
		Gson gson = new Gson();
		CurrentStateData state = gson.fromJson(json, CurrentStateData.class);
		Instance target = CurrentStateUtil.toInstance(state);
		
		try {
			double[] distrib = em.distributionForInstance(target);
			int maxCluster = EMCluster.findMaxCluster(distrib);
			EMCluster cluster = clusterToLabels.get(maxCluster);
			String label = cluster.getRingerLabel();
			double probOfCluster = distrib[maxCluster];
			double probOfLabel = cluster.getProbOfLabel();
			GMClassifyResponse gmResponse = new GMClassifyResponse(label, probOfLabel,
					probOfCluster);
			String respString = gson.toJson(gmResponse);
			response.getWriter().write(respString);
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
			byte[] encoded = Files.readAllBytes(Paths.get(
					DataUploadServlet.CLUSTER_LABELS_FILE));
			String json = Charset.defaultCharset().decode(
					ByteBuffer.wrap(encoded)).toString();
			
			Gson gson = new Gson();
			Type collectionType = new TypeToken<List<EMCluster>>(){}.getType();
			clusterToLabels = gson.fromJson(json, collectionType);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			throw new ServletException();
		} catch (Exception e) {
			e.printStackTrace();
			throw new ServletException();
		}

    }

}
