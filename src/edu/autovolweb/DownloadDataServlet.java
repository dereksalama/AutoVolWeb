package edu.autovolweb;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;

/**
 * Servlet implementation class DownloadDataServlet
 */
@WebServlet("/DownloadDataServlet")
public class DownloadDataServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;
 

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String userId = request.getParameter("user");
		String type = request.getParameter("type");
		Instances allData;
		if (type != null && type.equals("em_loc")) {
			try {
				allData = loadEmLocData(userId);
			} catch (Exception e) {
				response.sendError(HttpServletResponse.SC_BAD_REQUEST);
				e.printStackTrace();
				return;
			}
		} else if (type != null && type.equals("kmeans_loc")) {
			try {
				allData = loadKmeansLocData(userId);
			} catch (Exception e) {
				response.sendError(HttpServletResponse.SC_BAD_REQUEST);
				e.printStackTrace();
				return;
			}
		} else {
			allData = ViewDataServlet.loadAllData(userId, this);
		}
		ArffSaver saver = new ArffSaver();
		saver.setInstances(allData);
		String filename = "download.arff";
		saver.setFile(new File(filename));
		saver.writeBatch();
		
		
		response.setContentType("application/arff");
		response.setHeader("Content-disposition","attachment; filename=data.arff");
		
		OutputStream out = response.getOutputStream();
        FileInputStream in = new FileInputStream(filename);
        byte[] buffer = new byte[4096];
        int length;
        while ((length = in.read(buffer)) > 0){
           out.write(buffer, 0, length);
        }
        in.close();
        out.flush();
	}
	
	protected Instances loadEmLocData(String userId) throws Exception {
		Instances data = ViewDataServlet.loadAllData(userId, this);
		
		String filename = DataUploadServlet.constructUserFileName(userId, 
				EmLocKnnClassifyServlet.EM_CLASSIFIER_FILE);
		File f = new File(filename);
		if (!f.exists()) {
			throw new Exception("Loc EM not ready");
		}
		EM em = (EM) SerializationHelper.read(new FileInputStream(filename));
		
		ArrayList<String> clusterNames = new ArrayList<String>(em.numberOfClusters());
		for (int i = 0; i < em.numberOfClusters(); i++) {
			clusterNames.add(i + "");
		}
		
		return EmLocKnnClassifyServlet.replaceLocData(data, em, clusterNames);
	}
	
	protected Instances loadKmeansLocData(String userId) throws Exception {
		Instances allData = ViewDataServlet.loadAllData(userId, this);
		
		SimpleKMeans locClusterer = (SimpleKMeans) SerializationHelper.read(new FileInputStream(
				DataUploadServlet.constructUserFileName(userId,DataUploadServlet.LOC_CLUSTERER_FILE)));
		Gson gson = new Gson();
		String locJson = getFileJson(userId, DataUploadServlet.LOC_DATA_FILE);
		Type locCollectionType = new TypeToken<List<String>>(){}.getType();
		List<String> locClusters = gson.fromJson(locJson, locCollectionType);
		
		Instances allDataLoc = CurrentStateUtil.replaceLocationData(allData, 
				new int[] {2,3,4}, locClusters, 
				locClusterer.getAssignments());
		allDataLoc.setClass(allDataLoc.attribute("ringer"));

		return allDataLoc;
	}
	
	private String getFileJson(String userId, String filename) throws IOException {
		String fullFilename = DataUploadServlet.constructUserFileName(userId, filename);
		byte[] encoded = Files.readAllBytes(Paths.get(fullFilename));
		String json = Charset.defaultCharset().decode(
				ByteBuffer.wrap(encoded)).toString();
		return json;
	}
}
