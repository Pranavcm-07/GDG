import 'package:file_picker/file_picker.dart';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_application_1/screens/dashboard_screen.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';



class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  _UploadScreenState createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  List<File> selectedFiles = [];

  Future<void> pickFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        allowMultiple: true,
        type: FileType.custom,
        allowedExtensions: ['jpg','png','doc','pdf','txt']
      );

      if (result != null) {
        setState(() {
          selectedFiles.addAll(result.files.map((file) => File(file.path!)).toList());
        });
      } else {
       print("User canceled file picking.");
      }
    } catch (e) {
      print("File picking error: $e");
    }
  }

  Future<void> uploadFile() async {
    if (selectedFiles.isEmpty) {
      print("no files selected");
      return;
    }

    File firstFile = selectedFiles.first;
    String fileName = firstFile.path.split('/').last;

    String? apiUrl = dotenv.env['FASTAPI_URL'];
    String? apiKey = dotenv.env['FAST_API_KEY'];

    if (apiUrl == null || apiKey == null) {
    print("API URL or API Key is missing");
    return;
  }

    var request = http.MultipartRequest('POST',Uri.parse(apiUrl))
      ..headers.addAll({
        "access_token": apiKey
      })
      ..files.add(
      await http.MultipartFile.fromPath('file',firstFile.path)
    );

    var response = await request.send();

  if (response.statusCode == 200) {
    var responseBody = await response.stream.bytesToString();
    Map<String, dynamic> jsonResponse = json.decode(responseBody);

    // Navigate to Dashboard Screen with response data
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => DashboardScreen(data: jsonResponse["compliance_results"][0]),
      ),
    );
  } else {
    print("Error: ${response.statusCode}");
  }
  }

  void removeFile(int index){
    setState(() {
      selectedFiles.removeAt(index);
    });
  }

  void clearAll(){
    setState(() {
      selectedFiles.clear();
    });
  }

  @override
Widget build(BuildContext context) {
  return Scaffold(
    backgroundColor: Colors.white,
    appBar: AppBar(
      backgroundColor: Colors.white,
      elevation: 0,
      leading: IconButton(
        icon: Icon(Icons.arrow_back, color: Color.fromARGB(255, 51, 85, 255), size: 30),
        onPressed: () {
          Navigator.pop(context);
        },
      ),
    ),
    body: SingleChildScrollView(
      child: Container(
        color: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.start, // Aligns the text to the left
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                  "Upload Files!",
                  style: TextStyle(
                    color: Color.fromARGB(255, 51, 85, 255),
                    fontWeight: FontWeight.w900,
                    fontFamily: "Roboto",
                    fontSize: 30,
                  ),
                ),
                SizedBox(height: 5),
                Text(
                  "Select the files you want to upload.",
                  style: TextStyle(
                    color: Colors.black,
                    fontFamily: "Roboto",
                    fontSize: 14,
                    fontWeight: FontWeight.w600
                  ),
                )
                  ],
                )
              ],
            ),
            Padding(
              padding: const EdgeInsets.only(top: 10, bottom: 20),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  SvgPicture.asset(
                    'assets/images/upload_svg.svg',  
                    height: 400, 
                    width: 300,  
                  ),
                  Text("Upload File", style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
                  SizedBox(height: 10),
                  Text(
                    "Browse and choose the files\nyou want to upload",
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 16, color: Colors.grey),
                  ),
                ],
              ),
            ),
            // Upload Button
            Align(
              alignment: Alignment.bottomCenter,
              child: GestureDetector(
                onTap: pickFile,
                child: Container(
                  height: 60,
                  width: 60,
                  decoration: BoxDecoration(
                    color: Color.fromARGB(255, 51, 85, 255),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(Icons.add, size: 30, color: Colors.white),
                ),
              ),
            ),
            SizedBox(height: 20),
            // Selected Files Section inside the scrollable body
            selectedFiles.isEmpty
              ? SizedBox.shrink()
              : Container(
                  constraints: BoxConstraints(
                    minHeight: 100.0, 
                    maxHeight: 300.0, 
                  ),
                  color: Colors.white,
                  child: ListView.builder(
                    shrinkWrap: true,
                    itemCount: selectedFiles.length,
                    itemBuilder: (context, index) {
                      return Card(
                        margin: EdgeInsets.symmetric(vertical: 8),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(15), // Rounded border for the Card
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(15), // Ensures ListTile follows the border radius
                          child: ListTile(
                            tileColor: Colors.grey[200], // Background color of each selected item
                            leading: Icon(Icons.insert_drive_file, color: Color.fromARGB(255, 51, 85, 255)),
                            title: Text(
                              selectedFiles[index].path.split('/').last,
                              style: TextStyle(color: Colors.black),
                            ),
                            trailing: IconButton(
                              icon: Icon(Icons.delete, color: Colors.black),
                              onPressed: () => removeFile(index),
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ),
          SizedBox(height: 30),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: selectedFiles.isNotEmpty ? uploadFile : null, // Disabled when no files are selected
                style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.symmetric(horizontal: 30, vertical: 4), // Padding
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8), // Rounded corners
                  ),
                  backgroundColor: Color.fromARGB(255, 51, 85, 255), // Button background color
                  foregroundColor: Colors.white, // Text color
                  disabledBackgroundColor: Colors.grey[400], // Gray when disabled
                  disabledForegroundColor: Colors.white, // Text color when disabled
                ),
                child: Text("Send", style: TextStyle(fontSize: 20)),
              ),
            ),
            SizedBox(height: 30),
          ],
        ),
      ),
    ),
  );
}
}