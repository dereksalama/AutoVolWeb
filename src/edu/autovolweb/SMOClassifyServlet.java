package edu.autovolweb;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.Map.Entry;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.neighboursearch.KDTree;

import com.google.gson.JsonObject;

/**
 * Servlet implementation class SMOClassifyServlet
 */
@WebServlet("/SMOClassifyServlet")
public class SMOClassifyServlet extends HttpServlet {
	private static final long serialVersionUID = 4544453414238079731L;
	private Instances data;
	private SMO smo;
	private KDTree kd;
	
    @Override
    public void doGet(HttpServletRequest req, HttpServletResponse resp)
            throws IOException {
    	
    	Instance target = new DenseInstance(data.numAttributes());
    	target.setDataset(data);
    	
    	Map<String, String[]> attrMap = req.getParameterMap();
    	for (Entry<String, String[]> e : attrMap.entrySet()) {
    		Attribute attr = data.attribute(e.getKey());
    		if (attr == null) {
    			continue;
    		}
    		Double val = Double.valueOf(e.getValue()[0]);
    		target.setValue(attr, val);
    	}
    	
    	String smoResult = "err";
    	try {
			double classVal = smo.classifyInstance(target);
			smoResult = data.classAttribute().value((int) classVal);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
    	
    	JsonObject json = new JsonObject();
        resp.setContentType("text/plain");

		json.addProperty("smo", smoResult);
		
		String kdThree = kdClassify(target, 3);
		json.addProperty("k3", kdThree);
		
		String kdFive = kdClassify(target, 7);
		json.addProperty("k7", kdFive);
		
		
	    resp.getWriter().write(json.toString());


    }
    
    private String kdClassify(Instance target, int k) {
    	String result = "err";
		try {
			Instances neighbors = kd.kNearestNeighbours(target, 3);
			int[] classCounts = new int[data.classAttribute().numValues()];
			for (Instance n : neighbors) {
				classCounts[(int) Math.round(n.classValue())]++;
			}
			
			int maxIndex = 0;
			int maxCount = Integer.MIN_VALUE;
			for (int i = 0; i < data.classAttribute().numValues(); i++) {
				if (classCounts[i] > maxCount) {
					maxCount = classCounts[i];
					maxIndex = i;
				}
			}
			
			return data.classAttribute().value(maxIndex);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
    	return result;
    }
    
    @Override
    public void init() throws ServletException {
    	InputStream dataFileStream = getServletContext()
    			.getResourceAsStream("WEB-INF/resources/labeled_output.csv");

    	CSVLoader loader = new CSVLoader();
    	try {
			loader.setSource(dataFileStream);
			data = loader.getDataSet();
			data.setClassIndex(data.numAttributes() - 1);
			smo = new SMO();
			smo.buildClassifier(data);
			dataFileStream.close();
			
			kd = new KDTree();
			kd.setInstances(data);
		} catch (IOException e) {
			e.printStackTrace();
			throw new ServletException("Unable to load csvfile");
		} catch (Exception e) {
			e.printStackTrace();
			throw new ServletException("Error building classifier");
		}
    }


}
