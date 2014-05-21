package edu.autovolweb;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

/**
 * Servlet implementation class RfClassifyServlet
 */
@WebServlet("/RfClassifyServlet")
public class RfClassifyServlet extends ClearingHttpServlet {
	private static final long serialVersionUID = 1L;
	
	private Map<String, FilteredClassifier> rfMap;
	private Set<String> initializedUsers;
	
	@Override
	public void init() throws ServletException {
		rfMap = new ConcurrentHashMap<>();
		initializedUsers = Collections.synchronizedSet(new HashSet<String>());
	}


	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		try {
			String userId = request.getParameter("user");
			if (userId == null) {
				response.sendError(HttpServletResponse.SC_BAD_REQUEST);
				return;
			}
			saveAttempt(userId);
			if (!initializedUsers.contains(userId)) {
				try {
					initForUser(userId);
				} catch (Exception e) {
					e.printStackTrace();
					response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
					return;
				}
				
			}
			initializedUsers.add(userId);
			String input = request.getParameter("target");
			if (input == null) {
				response.sendError(HttpServletResponse.SC_BAD_REQUEST);
				return;
			}
			
			Instance target = constructTarget(input, userId);
			if (target == null) {
				response.sendError(HttpServletResponse.SC_BAD_REQUEST);
				return;
			}
			JsonObject json = new JsonObject();
			
			FilteredClassifier rf = rfMap.get(userId);
			double classification;
			try {
				classification = rf.classifyInstance(target);
				String classStr = target.classAttribute().value((int) classification);
				json.addProperty("result",classStr);
			} catch (Exception e) {
				e.printStackTrace();
				json.addProperty("result","err");
			}
			response.getWriter().write(json.toString());
			
			String disabled = request.getParameter("disabled");
			if (disabled == null) {
				disabled = "unknown";
			}
		
			saveResult(json.toString(), disabled, userId);
		} catch (IOException e) {
			e.printStackTrace();
			clear();
			throw e;
		} catch (Exception e) {
			e.printStackTrace();
			clear();
			throw e;
		}
	}
	
	protected Instance constructTarget(String input, String userId) {
		Gson gson = new Gson();
		CurrentStateData state = gson.fromJson(input, CurrentStateData.class);
		return CurrentStateUtil.toInstance(state);
	}
	
	protected void initForUser(String userId) throws Exception {
		Instances data = loadData(userId);
		RandomForest rf = new RandomForest();
		
		Remove r = new Remove();
		int[] attrIndices = new int[1];
		attrIndices[0] = data.attribute("day").index();
		r.setAttributeIndicesArray(attrIndices);
		r.setInputFormat(data);
		FilteredClassifier filtered = new FilteredClassifier();
		filtered.setClassifier(rf);
		filtered.setFilter(r);
		
		filtered.buildClassifier(data);
		
		rfMap.put(userId, filtered);
	}

	protected Instances loadData(String userId) throws Exception {
		Instances data = ViewDataServlet.loadAllData(userId, this);
		data.setClass(data.attribute("ringer"));
		return data;
	}
	
	protected Instances removeDay(Instances data) throws Exception {
		Remove r = new Remove();
		int[] attrIndices = new int[2];
		attrIndices[0] = data.attribute("day").index();
		attrIndices[1] = data.attribute("loc_provider").index();
		r.setAttributeIndicesArray(attrIndices);
		r.setInputFormat(data);
		return Filter.useFilter(data, r);
	}
	
	protected void saveAttempt(String userId) throws IOException {
		String filename = DataUploadServlet.constructUserFileName(userId,
				getClass().getSimpleName());
		BufferedWriter writer = new BufferedWriter(new FileWriter(filename, true));
		
		writer.write(new Date() + " --> attempt \n");
		writer.close();
	}
	
	protected void saveResult(String result, String disabled, String userId) throws IOException {
		String filename = DataUploadServlet.constructUserFileName(userId,
				getClass().getSimpleName());
		BufferedWriter writer = new BufferedWriter(new FileWriter(filename, true));
		
		writer.write(new Date() + " --> " + result + ", disabled: " + disabled + "\n");
		writer.close();
	}
	
	


	@Override
	protected void clear() {
		rfMap.clear();
		initializedUsers.clear();	
	}


}
