
//////////////////////////////////////////////////////////////////////////////////
////// Copyright (c) 2019-20, Lawrence Livermore National Security, LLC and XPlacer
////// project contributors. See the COPYRIGHT file for details.
//////
////// SPDX-License-Identifier: (BSD-3-Clause)
//////////////////////////////////////////////////////////////////////////////////

/// Converts flat CSV files (produced by the XPlacer/Tracer runtime) to a memory access map
/// 
/// \note This code requires the Cimg.h library
/// \author pirkelbauer2@llnl.gov

#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <cmath>

#define cimg_display 0
#include "CImg.h"

const char* const description =
  "csv2img [options] csv-file\n"
  "  creates an image file from a flat CSV file.\n"
  "  Use -h for help\n";

const char* const synopsis    =
  "csv2img [options] csv-file\n"
  "    -h      this help message\n"
  "  output options\n"
  "    -f ext  output format. If -o is specified -f is ignored\n"
  "    -o file output file\n"
  "    -p num  number of pixels (w,h) that an entry in the csv files will occupy [default: 16]\n"
  "    -w num  number of entries on a row [default: sqrt(|entries|)]\n"
  "  color options\n"
  "    -c num  red, green, and blue value of a dark pixel\n"
  "    -C num  red, green, and blue value of a light pixel\n"
  "    -r num  red component of a dark pixel\n"
  "    -R num  red component of a light pixel\n"
  "    -g num  green component of a dark pixel\n"
  "    -G num  green component of a light pixel\n"
  "    -b num  blue component of a dark pixel\n"
  "    -B num  blue component of a light pixel\n"
  ;


constexpr size_t PX = 16;

namespace cl = cimg_library;

typedef unsigned char color_value_t;

namespace channel_info
{
  const size_t red = 0;
  const size_t green = 1;
  const size_t blue = 2;
}

struct Settings
{
  size_t      px     = 16;
  size_t      w      = 0;
  size_t      r      = 0;
  size_t      R      = 255;
  size_t      g      = 0;
  size_t      G      = 255;
  size_t      b      = 0;
  size_t      B      = 255;
  std::string format = "png";
  std::string output = "";
};

/// A forward iterator over the image channels.
struct pixelator : std::iterator<std::forward_iterator_tag, color_value_t>
{
  pixelator(const Settings& settings, size_t w, color_value_t* red, color_value_t* green, color_value_t* blue)
  : pos(0), s(settings), rowlen(w/s.px), r(red), g(green), b(blue)
  {}

  /// writes a character to the output stream.
  pixelator& operator=(color_value_t value)
  {
    const size_t  w    = rowlen * s.px;
    const size_t  row  = pos / rowlen;
    const size_t  col  = pos % rowlen;
    const size_t  base = (row * w + col) * s.px;
    color_value_t rv   = value ? s.R : s.r;
    color_value_t gv   = value ? s.G : s.g;
    color_value_t bv   = value ? s.B : s.b;

    for (size_t i = 0; i < s.px; ++i)
    {
      size_t ofs = base + i * w;

      for (size_t j = 0; j < s.px; ++j)
      {
        r[ofs] = rv;
        g[ofs] = gv;
        b[ofs] = bv;

        ++ofs;
      }
    }

    return *this;
  }

  pixelator& operator*()     { return *this; }

  pixelator& operator++()
  {
    ++pos;
    return *this;
  }

  pixelator operator++(int)
  {
    pixelator tmp(*this);

    ++pos;
    return tmp;
  }

  private:
          size_t         pos;
    const Settings       s;
    const size_t         rowlen;
    color_value_t* const r;
    color_value_t* const g;
    color_value_t* const b;
};


void convert(const std::vector<color_value_t>& data, color_value_t* out, size_t w, size_t h, size_t d, size_t /*s*/, const Settings& s)
{
  assert(d == 1);
  assert(s.w == 0 || w == s.w * s.px);

  const size_t   len         = w * h;
  color_value_t* red_channel = out + len * channel_info::red;
  color_value_t* gre_channel = out + len * channel_info::green;
  color_value_t* blu_channel = out + len * channel_info::blue;

  std::copy(data.begin(), data.end(), pixelator(s, w, red_channel, gre_channel, blu_channel));
}


void convert(const std::vector<color_value_t>& data, cl::CImg<color_value_t>& out, const Settings& settings)
{
  convert(data, out.data(), out.width(), out.height(), out.depth(), out.spectrum(), settings);
}

std::vector<color_value_t> readCsv(std::string fn)
{
  std::vector<color_value_t> data;
  std::ifstream              csv(fn);

  if (!csv.is_open())
  {
    fn.append(" <> file not found");
    throw std::logic_error(fn);
  }

  int val;

  csv >> val;
  while (!csv.eof())
  {
    data.push_back(val);

    char c;

    csv >> c;
    assert(c == ',');

    csv >> val;
  }

  return data;
}

std::pair<size_t, size_t>
computeDimensions(std::vector<color_value_t>& data, Settings& settings)
{
  const size_t n = data.size();
  const size_t w = (settings.w == 0) ? std::sqrt(n) : settings.w;
  const size_t h = ((n+w-1)) / w;

  return std::make_pair(w * settings.px, h * settings.px);
}

template <class U, class V>
U conv(const V& val)
{
  std::stringstream tmp;
  U                 res;

  tmp << val;
  tmp >> res;
  return res;
}

template <class N, class T>
bool matchOpt1(const std::vector<std::string>& args, N& pos, std::string opt, T& fld)
{
  std::string arg(args.at(pos));
  
  if (arg.find(opt) != 0) return false;

  ++pos;
  fld = conv<T>(args.at(pos));
  ++pos;
  return true;
}

template <class N, class Fn>
bool matchOpt1(const std::vector<std::string>& args, N& pos, std::string opt, Fn setter, Settings& s)
{
  std::string arg(args.at(pos));

  if (arg.find(opt) != 0) return false;

  ++pos;
  setter(s, args.at(pos));
  ++pos;
  return true;
}

template <class N, class Fn>
bool matchOpt0(const std::vector<std::string>& args, N& pos, std::string opt, Fn fn)
{
  std::string arg(args.at(pos));  

  if (arg.find(opt) != 0) return false;

  fn();
  ++pos;
  return true;
}

void setColors(Settings& s, std::string val)
{
  s.r = s.g = s.b = conv<size_t>(val);
}

void SetColors(Settings& s, std::string val)
{
  s.R = s.G = s.B = conv<size_t>(val);
}

void help()
{
  std::cerr << synopsis << std::endl;
  exit(0);
}

void missingCsvFile()
{
  std::cerr << "Error: missing input file" << std::endl;
  std::cerr << "csv2img srcfile" << std::endl;
  std::cerr << "  use -h for help" << std::endl;
}

void batchModeIncompatible()
{
  std::cerr << "Error: multiple input files specified and -o present" << std::endl;
  std::cerr << "csv2img srcfile" << std::endl;
  std::cerr << "  use -h for help" << std::endl;
}

int main(int argc, char** argv)
{
  constexpr bool firstMatch = false;
  
  if (argc < 2)
  {
	  std::cerr << description << std::endl;
    return 1;
  }
  
  std::vector<std::string>   arguments(argv, argv+argc);
  size_t                     argn = 1;
  Settings                   settings;
  bool                       matched = true;

  while (matched)
  {
    matched = firstMatch
              || matchOpt1(arguments, argn, "-p", settings.px)
              || matchOpt1(arguments, argn, "-w", settings.w)
              || matchOpt1(arguments, argn, "-r", settings.r)
              || matchOpt1(arguments, argn, "-R", settings.R)
              || matchOpt1(arguments, argn, "-g", settings.g)
              || matchOpt1(arguments, argn, "-G", settings.G)
              || matchOpt1(arguments, argn, "-b", settings.b)
              || matchOpt1(arguments, argn, "-B", settings.B)
              || matchOpt1(arguments, argn, "-c", setColors, settings)
              || matchOpt1(arguments, argn, "-C", SetColors, settings)
              || matchOpt1(arguments, argn, "-f", settings.format)
              || matchOpt1(arguments, argn, "-o", settings.output)
              || matchOpt0(arguments, argn, "-h", help)
              ;
  }
 
  if (argn == arguments.size())
    missingCsvFile();
  
  if ((argn > arguments.size()-1) && (settings.output.size() > 0))
    batchModeIncompatible();
  
  //~ std::cerr << settings.r << "/" << settings.g << "/" << settings.b << std::endl;
  //~ std::cerr << settings.R << "/" << settings.G << "/" << settings.B << std::endl;

  while (argn < arguments.size())
  {
    const std::string          origfile(arguments.at(argn));
    std::cerr << origfile
              << " " << argn 
              << " " << argc
              << std::endl; 

    std::vector<color_value_t> data = readCsv(origfile);
    std::pair<size_t, size_t>  dims = computeDimensions(data, settings);
  
    std::cerr << origfile 
              << ": read " << data.size() << " values = " 
              << dims.first << "x" << dims.second 
              << std::endl
              ;
    
    cl::CImg<color_value_t>    img(dims.first, dims.second, 1 /* single frame */, 3 /* channels */, 0);
  
    convert(data, img, settings);
    
    std::string                outname = settings.output;
  
    if (outname.size() == 0)
    {
      const size_t             extpos = origfile.find_last_of(".");
      assert(extpos != 0 && extpos != std::string::npos);
  
      outname = origfile.substr(0, extpos) + "." + settings.format;
    }
  
    img.save(outname.c_str());
    ++argn;
  }
  
  return 0;
}
