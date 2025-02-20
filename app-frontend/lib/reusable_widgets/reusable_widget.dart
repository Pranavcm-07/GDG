// // import 'package:flutter/material.dart';

// // Image logoWidget(String imageName) {
// //   return Image.asset(
// //     imageName,
// //     fit: BoxFit.fitWidth,
// //     width: 240,
// //     height: 240,
// //     color: Colors.white,
// //   );
// // }

// // TextField reusableTextField(String text, IconData icon, bool isPasswordType,
// //     TextEditingController controller) {
// //   return TextField(
// //     controller: controller,
// //     obscureText: isPasswordType,
// //     enableSuggestions: !isPasswordType,
// //     autocorrect: !isPasswordType,
// //     cursorColor: Colors.white,
// //     style: TextStyle(color: Colors.white.withOpacity(0.9)),
// //     decoration: InputDecoration(
// //       prefixIcon: Icon(
// //         icon,
// //         color: Colors.white70,
// //       ),
// //       labelText: text,
// //       labelStyle: TextStyle(color: Colors.white.withOpacity(0.9)),
// //       filled: true,
// //       floatingLabelBehavior: FloatingLabelBehavior.never,
// //       fillColor: Colors.white.withOpacity(0.3),
// //       border: OutlineInputBorder(
// //           borderRadius: BorderRadius.circular(30.0),
// //           borderSide: const BorderSide(width: 0, style: BorderStyle.none)),
// //     ),
// //     keyboardType: isPasswordType
// //         ? TextInputType.visiblePassword
// //         : TextInputType.emailAddress,
// //   );
// // }

// // Container firebaseUIButton(BuildContext context, String title, Function onTap) {
// //   return Container(
// //     width: MediaQuery.of(context).size.width,
// //     height: 50,
// //     margin: const EdgeInsets.fromLTRB(0, 10, 0, 20),
// //     decoration: BoxDecoration(borderRadius: BorderRadius.circular(90)),
// //     child: ElevatedButton(
// //       onPressed: () {
// //         onTap();
// //       },
// //       style: ButtonStyle(
// //           backgroundColor: WidgetStateProperty.resolveWith((states) {
// //             if (states.contains(WidgetState.pressed)) {
// //               return Colors.black26;
// //             }
// //             return Colors.white;
// //           }),
// //           shape: WidgetStateProperty.all<RoundedRectangleBorder>(
// //               RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)))),
// //       child: Text(
// //         title,
// //         style: const TextStyle(
// //             color: Colors.black87, fontWeight: FontWeight.bold, fontSize: 16),
// //       ),
// //     ),
// //   );
// // }

// import 'package:flutter/material.dart';

// Image logoWidget(String imageName) {
//   return Image.asset(
//     imageName,
//     fit: BoxFit.fitWidth,
//     width: 240,
//     height: 240,
//     color: Colors.white,
//   );
// }

// TextField reusableTextField(String text, IconData icon, bool isPasswordType,
//     TextEditingController controller) {
//   return TextField(
//     controller: controller,
//     obscureText: isPasswordType,
//     enableSuggestions: !isPasswordType,
//     autocorrect: !isPasswordType,
//     cursorColor: Colors.white,
//     style: TextStyle(color: Colors.white.withOpacity(0.9)),
//     decoration: InputDecoration(
//       prefixIcon: Icon(
//         icon,
//         color: Colors.white70,
//       ),
//       labelText: text,
//       labelStyle: TextStyle(color: Colors.white.withOpacity(0.9)),
//       filled: true,
//       floatingLabelBehavior: FloatingLabelBehavior.never,
//       fillColor: Colors.white.withOpacity(0.3),
//       border: OutlineInputBorder(
//           borderRadius: BorderRadius.circular(30.0),
//           borderSide: const BorderSide(width: 0, style: BorderStyle.none)),
//     ),
//     keyboardType: isPasswordType
//         ? TextInputType.visiblePassword
//         : TextInputType.emailAddress,
//   );
// }

// Container firebaseUIButton(BuildContext context, String title, Function onTap) {
//   return Container(
//     width: MediaQuery.of(context).size.width,
//     height: 50,
//     margin: const EdgeInsets.fromLTRB(0, 10, 0, 20),
//     decoration: BoxDecoration(borderRadius: BorderRadius.circular(90)),
//     child: ElevatedButton(
//       onPressed: () {
//         onTap();
//       },
//       style: ButtonStyle(
//           backgroundColor: MaterialStateProperty.all(Colors.white),
//           foregroundColor: MaterialStateProperty.all(Colors.black87),
//           shape: MaterialStateProperty.all<RoundedRectangleBorder>(
//               RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)))),
//       child: Text(
//         title,
//         style: const TextStyle(
//             color: Colors.black87, fontWeight: FontWeight.bold, fontSize: 16),
//       ),
//     ),
//   );
// }


import 'package:flutter/material.dart';

Image logoWidget(String imageName) {
  return Image.asset(
    imageName,
    fit: BoxFit.fitWidth,
    width: 350,
    height: 250,
    
  );
}

// TextField reusableTextField(String text, IconData icon, bool isPasswordType,
//     TextEditingController controller) {
//   return TextField(
//     controller: controller,
//     obscureText: isPasswordType,
//     enableSuggestions: !isPasswordType,
//     autocorrect: !isPasswordType,
//     cursorColor: Colors.white,
//     style: TextStyle(color: Colors.white.withOpacity(0.9)),
//     decoration: InputDecoration(
//       prefixIcon: Icon(
//         icon,
//         color: Colors.white70,
//       ),
//       labelText: text,
//       labelStyle: TextStyle(color: Colors.white.withOpacity(0.9)),
//       filled: true,
//       floatingLabelBehavior: FloatingLabelBehavior.never,
//       fillColor: Colors.white.withOpacity(0.3),
//       border: OutlineInputBorder(
//           borderRadius: BorderRadius.circular(30.0),
//           borderSide: const BorderSide(width: 0, style: BorderStyle.none)),
//     ),
//     keyboardType: isPasswordType
//         ? TextInputType.visiblePassword
//         : TextInputType.emailAddress,
//   );
// }

TextField reusableTextField(String text, IconData icon, bool isPasswordType,
    TextEditingController controller) {
  return TextField(
    controller: controller,
    obscureText: isPasswordType,
    enableSuggestions: !isPasswordType,
    autocorrect: !isPasswordType,
    cursorColor: Color.fromARGB(255, 51, 85, 255), // Blue cursor
    style: TextStyle(color: Colors.black), // Black text
    decoration: InputDecoration(
      prefixIcon: Icon(
        icon,
        color: Color.fromARGB(255, 51, 85, 255), // Blue icon
      ),
      labelText: text,
      labelStyle: TextStyle(color: Colors.black54), // Dark gray label
      filled: true,
      floatingLabelBehavior: FloatingLabelBehavior.never,
      fillColor: Colors.grey[200], // Light gray background
      border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10.0),
          borderSide: BorderSide.none), // No border
    ),
    keyboardType: isPasswordType
        ? TextInputType.visiblePassword
        : TextInputType.emailAddress,
  );
}



// Container firebaseUIButton(BuildContext context, String title, Function onTap) {
//   return Container(
//     width: MediaQuery.of(context).size.width,
//     height: 50,
//     margin: const EdgeInsets.fromLTRB(0, 10, 0, 20),
//     decoration: BoxDecoration(borderRadius: BorderRadius.circular(90)),
//     child: ElevatedButton(
//       onPressed: () {
//         onTap();
//       },
//       style: ButtonStyle(
//           backgroundColor: MaterialStateProperty.all(Colors.white),
//           foregroundColor: MaterialStateProperty.all(Colors.black87),
//           shape: MaterialStateProperty.all<RoundedRectangleBorder>(
//               RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)))),
//       child: Text(
//         title,
//         style: const TextStyle(
//             color: Colors.black87, fontWeight: FontWeight.bold, fontSize: 16),
//       ),
//     ),
//   );
// }
Container firebaseUIButton(BuildContext context, String title, Function onTap) {
  return Container(
    width: MediaQuery.of(context).size.width,
    height: 50,
    margin: const EdgeInsets.fromLTRB(0, 10, 0, 20),
    decoration: BoxDecoration(borderRadius: BorderRadius.circular(90)),
    child: ElevatedButton(
      onPressed: () {
        onTap();
      },
      style: ButtonStyle(
          backgroundColor: MaterialStateProperty.all(Color.fromARGB(255, 51, 85, 255)), // Blue button
          foregroundColor: MaterialStateProperty.all(Colors.white), // White text
          shape: MaterialStateProperty.all<RoundedRectangleBorder>(
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)))),
      child: Text(
        title,
        style: const TextStyle(
            color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16),
      ),
    ),
  );
}